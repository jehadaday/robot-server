
"""Robot server: raw 8 kHz PCM from the ESP32 -> Gemini (hears, wake word, reply) -> edge-tts -> 16 kHz PCM back."""
import asyncio
import io
import os
import struct
import threading
import time
 
import edge_tts
import miniaudio
import requests
from flask import Flask, Response, request
 
API_KEY = os.environ.get("GEMINI_API_KEY", "")
ROBOT_TOKEN = os.environ.get("ROBOT_TOKEN", "")          # shared secret with the robot (set it on Render)
MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]    # tried in order if one is rate-limited
VOICE = "ar-IQ-BasselNeural"                              # Iraqi male voice (ar-IQ-RanaNeural = female)
IN_RATE = 8000                                            # what the robot sends
OUT_RATE = 16000                                          # what the robot plays
NAME = "روبوت"
 
SYSTEM = f"""اسمك {NAME}، وانت روبوت صغير ظريف يحجي باللهجة العراقية البسيطة. المستخدم يحجي وياك بالصوت.
اكتب اولا ما سمعته من المستخدم بالضبط كنص، بعدها اكتب ||| ، بعدها ردك.
الشكل: <ما سمعته> ||| <ردك>
اذا ما في كلام واضح، اكتب فقط كلمة IGNORE.
ردك يكون جملة او جملتين قصار فقط، بدون رموز او ايموجي او قوائم، لأنه راح ينقال بالصوت."""
 
TRANSCRIBE = "اكتب بالضبط ما قيل في هذا التسجيل الصوتي نصاً فقط، بدون اي اضافة. اذا ما في كلام بشري واضح اكتب فقط كلمة IGNORE."
TEXT_SYSTEM = (f"اسمك {NAME}، وانت روبوت صغير ظريف يحجي باللهجة العراقية البسيطة. "
               "ردك جملة او جملتين قصار فقط، بدون رموز او ايموجي او قوائم، لأنه راح ينقال بالصوت. "
               "ابدأ ردك دائماً بوسم واحد بين اقواس مربعة يعبر عن شعورك بالرد، واحد من: [happy] [sad] [surprised] [love] [star] [wink] [neutral]. "
               "مثلاً: [happy] هلا بيك! استخدم [love] للمحبة والشكر، [star] للاعجاب او الانجاز، [wink] للمزاح.")
NAME_FORMS = ["روبوت", "روبت", "ربوت", "رووبوت", "روبووت", "روبوط", "روبو"]
AWAKE_WINDOW = 14   # seconds the robot keeps listening after "نعم"
 
app = Flask(__name__)
history = []
awake_until = 0.0
lock = threading.Lock()
cache = {}
 
 
def wav_from_pcm(pcm: bytes, rate: int) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm
 
 
def _gemini(body):
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": API_KEY}, json=body, timeout=40)
            if r.status_code == 200:
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            print("gemini", model, r.status_code, r.text[:200])
        except Exception as e:
            print("gemini error", model, e)
    return None
 
 
def _norm(t):
    import re
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^\w\s]", " ", t).strip()
 
 
def has_name(heard):
    words = _norm(heard).split()
    return any(w.lstrip("يوها") in NAME_FORMS or w in NAME_FORMS or w.lstrip("و") in NAME_FORMS or "روبوت" in w for w in words)
 
 
def only_name(heard):
    words = [w for w in _norm(heard).split() if not (w in ("يا", "هلا", "اهلا", "هاي", "مرحبا", "هلو", "الو", "اي") or has_name(w))]
    return len(" ".join(words)) < 4
 
 
def transcribe(pcm):
    import base64
    audio = {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_from_pcm(pcm, IN_RATE)).decode()}}
    body = {"contents": [{"role": "user", "parts": [{"text": TRANSCRIBE}, audio]}], "generationConfig": {"maxOutputTokens": 150}}
    t = _gemini(body)
    print("transcript:", t)
    if not t or "IGNORE" in t:
        return None
    return t.strip()
 
 
MOODS = {"happy", "sad", "surprised", "love", "star", "wink", "neutral"}
NUM_WORDS = ["", "واحد", "اثنين", "ثلاثة", "اربعة", "خمسة", "ستة"]
 
 
game_on = False
RPS_NAMES = ["حجرة", "ورقة", "مقص"]
RPS_KEYS = ["حجر", "ورق", "مقص"]
 
 
def respond(heard):
    """Returns (reply, mood, extra_headers, keep_awake)."""
    global game_on
    import random, re
    n = _norm(heard)
    words = set(n.split())
    nl = n.lower()
    if ("واي فاي" in n or "وايفاي" in n or "wifi" in nl or "الشبكه" in n or "الانترنت" in n) and \
            any(w in n for w in ("غير", "تغيير", "اعداد", "بدل")):
        game_on = False
        return "تمام، راح افتح وضع الاعداد. اتصل من تلفونك بشبكة روبوت سيت اب", "happy", {"X-Wifi": "1"}, False
    found = [i for i, k in enumerate(RPS_KEYS) if k in n]
    if game_on and words & {"كفايه", "كافي", "خلص", "بس", "وقف", "انتهت", "انهي", "كفى"}:
        game_on = False
        return "تمام، انتهت اللعبة", "happy", {}, False
    if not game_on and (len(found) == 3 or (any(w in n for w in ("نلعب", "لعبه", "العب")) and found)):
        game_on = True
        return "تمام! اختار حجرة او ورقة او مقص", "happy", {}, True
    if game_on:
        if len(found) == 1:
            u = found[0]
            r = random.randint(0, 2)
            diff = (u - r) % 3
            if diff == 0:
                res, mood = "تعادلنا", "wink"
            elif diff == 1:
                res, mood = "انت فزت، مبروك", "sad"
            else:
                res, mood = "انا فزت", "happy"
            return f"انا اخترت {RPS_NAMES[r]}. {res}", mood, {"X-Rps": f"{u},{r}"}, True
        return "ما فهمت، قل حجرة او ورقة او مقص", "neutral", {}, True
    if "نرد" in n or "زهر" in n or "dice" in nl:
        d = random.randint(1, 6)
        return f"رميت النرد وطلع الرقم {NUM_WORDS[d]}", "happy", {"X-Dice": str(d)}, False
    t = answer_text(heard)
    m = re.match(r"\s*\[(\w+)\]\s*", t)
    mood = "neutral"
    if m:
        if m.group(1).lower() in MOODS:
            mood = m.group(1).lower()
        t = t[m.end():]
    t = re.sub(r"\[\w+\]", "", t).strip()
    return (t or "تمام"), mood, {}, False
 
 
def answer_text(heard):
    contents = list(history) + [{"role": "user", "parts": [{"text": heard}]}]
    body = {"systemInstruction": {"parts": [{"text": TEXT_SYSTEM}]}, "contents": contents, "generationConfig": {"maxOutputTokens": 300}}
    t = _gemini(body)
    return t[:300] if t else "صار عندي خطأ بالاتصال، جرب مرة ثانية."
 
 
async def _tts(text: str) -> bytes:
    mp3 = b""
    async for part in edge_tts.Communicate(text, VOICE).stream():
        if part["type"] == "audio":
            mp3 += part["data"]
    return mp3
 
 
def speak_pcm(text: str) -> bytes:
    mp3 = asyncio.run(_tts(text))
    dec = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=OUT_RATE)
    return dec.samples.tobytes()
 
 
def authorized() -> bool:
    return (not ROBOT_TOKEN) or request.headers.get("X-Robot-Key") == ROBOT_TOKEN
 
 
@app.get("/ping")
def ping():
    return ("ok", 200) if authorized() else ("no", 403)
 
 
@app.get("/hello")
def hello():
    if not authorized():
        return ("no", 403)
    if "hello" not in cache:
        cache["hello"] = speak_pcm("تم الاتصال، أنا جاهز في خدمتك.")
    return Response(cache["hello"], mimetype="application/octet-stream")
 
 
@app.post("/talk")
def talk():
    global awake_until, game_on
    if not authorized():
        return ("no", 403)
    pcm = request.get_data()
    if len(pcm) < 4000:
        return ("", 204)
    with lock:
        heard = transcribe(pcm)
        if not heard:
            return ("", 204)
        awake = time.time() < awake_until
        if not awake:
            game_on = False
            if not has_name(heard):
                print("ignored:", heard)
                return ("", 204)
            if only_name(heard):
                awake_until = time.time() + AWAKE_WINDOW
                if "yes" not in cache:
                    cache["yes"] = speak_pcm("نعم، اسمعك")
                print("wake only ->", heard)
                return Response(cache["yes"], mimetype="application/octet-stream", headers={"X-Awake": "1"})
        reply, mood, headers, keep = respond(heard)
        headers = dict(headers)
        headers["X-Mood"] = mood
        if keep:
            awake_until = time.time() + AWAKE_WINDOW
            headers["X-Awake"] = "1"
        else:
            awake_until = 0
        print("heard:", heard, "| reply:", reply, "|", headers)
        history.append({"role": "user", "parts": [{"text": heard}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        del history[:-12]
    try:
        out = speak_pcm(reply)
    except Exception as e:
        print("tts error", e)
        return ("", 500)
    return Response(out, mimetype="application/octet-stream", headers=headers)
 
 
@app.get("/")
def index():
    return "robot server running"
 
