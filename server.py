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
CHAT_WINDOW = 12                                          # seconds after a reply when no wake word is needed
NAME = "روبوت"

SYSTEM = f"""اسمك {NAME}، وانت روبوت صغير ظريف يحجي باللهجة العراقية البسيطة. المستخدم يحجي وياك بالصوت.
اكتب اولا ما سمعته من المستخدم بالضبط كنص، بعدها اكتب ||| ، بعدها ردك.
الشكل: <ما سمعته> ||| <ردك>
{{MODE}}
اذا ما في كلام واضح، اكتب فقط كلمة IGNORE.
ردك يكون جملة او جملتين قصار فقط، بدون رموز او ايموجي او قوائم، لأنه راح ينقال بالصوت."""

WAKE_MODE = (f"لا ترد الا اذا المستخدم ناداك باسمك ({NAME}، مثل: يا {NAME}، {NAME}، هلا {NAME}). "
             "اذا ما ناداك باسمك، او الكلام مو موجه الك، اكتب فقط كلمة IGNORE. "
             "اذا ناداك باسمك بس بدون سؤال، رد: نعم، اسمعك.")
CHAT_MODE = "المستخدم يكمل الحديث وياك، فجاوبه مباشرة بدون ما يحتاج يناديك. اذا الكلام مو موجه الك اكتب IGNORE."

app = Flask(__name__)
history = []
last_reply_at = 0.0
lock = threading.Lock()
cache = {}


def wav_from_pcm(pcm: bytes, rate: int) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def ask_gemini(pcm: bytes):
    """Returns (heard, reply) or None when the robot should stay silent."""
    import base64
    global last_reply_at
    in_chat = (time.time() - last_reply_at) < CHAT_WINDOW
    system = SYSTEM.replace("{MODE}", CHAT_MODE if in_chat else WAKE_MODE)
    audio = {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_from_pcm(pcm, IN_RATE)).decode()}}
    contents = list(history) + [{"role": "user", "parts": [audio]}]
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"maxOutputTokens": 300}}
    text = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": API_KEY}, json=body, timeout=40)
            if r.status_code == 200:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                break
            print("gemini", model, r.status_code, r.text[:200])
        except Exception as e:
            print("gemini error", model, e)
    if text is None:
        return ("", "صار عندي خطأ بالاتصال، جرب مرة ثانية.")
    print("gemini:", text)
    if not text or "IGNORE" in text:
        return None
    if "|||" in text:
        heard, reply = [t.strip() for t in text.split("|||", 1)]
    else:
        heard, reply = "", text
    if not reply or "IGNORE" in reply:
        return None
    return heard, reply[:300]


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
    global last_reply_at
    if not authorized():
        return ("no", 403)
    pcm = request.get_data()
    if len(pcm) < 4000:
        return ("", 204)
    with lock:
        result = ask_gemini(pcm)
        if result is None:
            return ("", 204)
        heard, reply = result
        print("heard:", heard, "| reply:", reply)
        if heard:
            history.append({"role": "user", "parts": [{"text": heard}]})
            history.append({"role": "model", "parts": [{"text": reply}]})
            del history[:-12]
    try:
        out = speak_pcm(reply)
    except Exception as e:
        print("tts error", e)
        return ("", 500)
    last_reply_at = time.time()
    return Response(out, mimetype="application/octet-stream")


@app.get("/")
def index():
    return "robot server running"
