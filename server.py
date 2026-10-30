"""University information-desk robot server.
ESP32 sends raw 8 kHz PCM -> Gemini transcribes -> wake name check -> Gemini picks the matching question
from the Google Sheet -> cached edge-tts voice + Arabic text pages (for the screen) go back to the robot."""
import array
import asyncio
import csv
import difflib
import hashlib
import io
import json
import math
import os
import re
import struct
import threading
import time
from collections import OrderedDict

import edge_tts
import miniaudio
import requests
from flask import Flask, Response, request

import arabic_text

API_KEY = os.environ.get("GEMINI_API_KEY", "")
ROBOT_TOKEN = os.environ.get("ROBOT_TOKEN", "")          # shared secret with the robot (set it on Render)
SHEET_URL = os.environ.get("SHEET_CSV_URL", "")          # Google Sheet link (published CSV or normal share link)
SHEET_REFRESH = 60                                       # seconds between sheet re-reads
MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]    # tried in order if one is rate-limited
VOICE = "ar-IQ-BasselNeural"                              # Iraqi male voice (ar-IQ-RanaNeural = female)
IN_RATE = 8000                                            # what the robot sends
OUT_RATE = 16000                                          # what the robot plays
VOICE_DRIVE = 2.0                                         # voice loudness (1 = as is, 2 = louder, 3 = very loud)
AWAKE_WINDOW = 14                                         # seconds the robot keeps listening after "نعم"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_SHEET = os.path.join(HERE, "sheet_sample.csv")      # used until/unless the online sheet can be read

TRANSCRIBE = "اكتب بالضبط ما قيل في هذا التسجيل الصوتي نصاً فقط، بدون اي اضافة. اذا ما في كلام بشري واضح اكتب فقط كلمة IGNORE."
MOODS = {"happy", "sad", "surprised", "love", "star", "wink", "neutral"}
LEGACY_NAME_FORMS = ["روبوت", "روبت", "ربوت", "رووبوت", "روبووت", "روبوط", "روبو"]
GREETING_WORDS = {"يا", "هلا", "اهلا", "هاي", "مرحبا", "هلو", "الو", "اي", "سلام", "السلام", "عليكم"}

app = Flask(__name__)
lock = threading.Lock()
awake_until = 0.0


# ---------------------------------------------------------------- text helpers
def _norm(t):
    t = re.sub(r"[ً-ْـ]", "", t or "")
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[^\w\s]", " ", t).strip().lower()


def wav_from_pcm(pcm: bytes, rate: int) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


# ---------------------------------------------------------------- Google Sheet knowledge
class KB:
    university = ""
    names = ["روبوت"]
    greeting = ""
    voice = ""        # sheet override: صوت (رنا / باسل)
    rate = ""         # sheet override: سرعة الصوت
    drive = ""        # sheet override: قوة الصوت (loudness 0.8 - 3)
    prons = []        # sheet: نطق rows -> [(written, spoken)]
    items = []        # {"id","questions","spoken","screen","mood","norm"}
    digest = ""
    loaded_at = 0.0
    error = ""
    source = ""


kb = KB()


def sheet_url(u):
    u = (u or "").strip()
    if not u:
        return ""
    if "output=csv" in u or "format=csv" in u or "/pub?" in u and "csv" in u:
        return u
    m = re.search(r"/spreadsheets/d/([\w-]+)", u)
    if m and "/d/e/" not in u:
        g = re.search(r"gid=(\d+)", u)
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={g.group(1) if g else 0}"
    return u


def split_questions(cell):
    return [q.strip() for q in re.split(r"[/|\n;؛]+", cell or "") if q.strip()]


def parse_sheet(text):
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    uni, names, greeting, items = "", [], "", []
    st = {"voice": "", "rate": "", "drive": "", "prons": []}
    for r in rows:
        r = [c.strip() for c in r]
        if not r or not r[0]:
            continue
        key = _norm(r[0])
        val = r[1] if len(r) > 1 else ""
        if key.startswith("#"):
            continue
        if key.startswith("اسم الجامعه") or key == "الجامعه":
            uni = "" if ("اكتب" in val or "..." in val) else val
            continue
        if key.startswith("اسم النداء") or key == "النداء" or key.startswith("اسم الروبوت"):
            names = [n.strip() for n in re.split(r"[/|،,;؛]+", val) if n.strip()]
            continue
        if key.startswith("رساله الترحيب") or key == "الترحيب":
            greeting = val
            continue
        if key == "صوت" or key.startswith("صوت الروبوت"):
            st["voice"] = val
            continue
        if key.startswith("قوه الصوت") or key.startswith("مستوي الصوت") or key.startswith("علو الصوت"):
            st["drive"] = val
            continue
        if key.startswith("سرعه") or key == "السرعه":
            st["rate"] = val
            continue
        if key == "نطق" or key.startswith("تصحيح النطق"):
            tgt = r[2] if len(r) > 2 else ""
            if val and tgt:
                st["prons"].append((val, tgt))
            continue
        if key.startswith("الاسئله") or key in ("السؤال", "سؤال"):
            continue          # header row
        spoken = val
        if not spoken:
            continue
        screen = r[2] if len(r) > 2 and r[2] else spoken
        mood = r[3].strip().lower() if len(r) > 3 and r[3].strip().lower() in MOODS else "neutral"
        qs = split_questions(r[0])
        if qs:
            items.append({"id": len(items) + 1, "questions": qs, "spoken": spoken, "screen": screen, "mood": mood,
                          "norm": [_norm(q) for q in qs]})
    parse_sheet.settings = st
    return uni, names, greeting, items


def load_sheet(force=False):
    """Read the sheet (online if SHEET_CSV_URL is set, otherwise the bundled sample)."""
    url = sheet_url(SHEET_URL)
    text, src = None, ""
    if url:
        try:
            r = requests.get(url + ("&" if "?" in url else "?") + f"_cb={int(time.time())}", headers={"Cache-Control": "no-cache"}, timeout=12)      # cache-buster: Google may serve a stale copy
            r.encoding = "utf-8"
            if r.status_code == 200 and "<html" not in r.text[:300].lower():
                text, src = r.text, "online"
            else:
                kb.error = f"sheet http {r.status_code} (is it shared as 'anyone with the link'?)"
        except Exception as e:
            kb.error = f"sheet error {e}"
    if text is None and not kb.items:
        try:
            with open(LOCAL_SHEET, encoding="utf-8") as f:
                text, src = f.read(), "sample file"
        except Exception as e:
            kb.error += f" | no sample: {e}"
    if text is None:
        return
    d = hashlib.md5(text.encode()).hexdigest()
    if d == kb.digest and not force:
        kb.loaded_at = time.time()
        return
    uni, names, greeting, items = parse_sheet(text)
    kb.university, kb.names, kb.greeting, kb.items = uni, names or ["روبوت"], greeting, items
    st = getattr(parse_sheet, "settings", {})
    kb.voice, kb.rate, kb.prons = st.get("voice", ""), st.get("rate", ""), st.get("prons", [])
    kb.drive = st.get("drive", "")
    kb.digest, kb.loaded_at, kb.source = d, time.time(), src
    if src == "online":
        kb.error = ""
    print(f"sheet loaded ({src}): {len(items)} questions, university={uni!r}, names={kb.names}")
    threading.Thread(target=prewarm, daemon=True).start()


def sheet_loop():
    while True:
        time.sleep(SHEET_REFRESH)
        try:
            load_sheet()
        except Exception as e:
            print("sheet loop error", e)


# ---------------------------------------------------------------- Gemini
_cool = {}          # model -> time until we skip it (quota exceeded / not available)


def _gemini(body):
    for model in MODELS:
        if _cool.get(model, 0) > time.time():
            continue
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            fast = json.loads(json.dumps(body))
            fast.setdefault("generationConfig", {})["thinkingConfig"] = {"thinkingBudget": 0}     # no "thinking" = faster
            r = requests.post(url, headers={"x-goog-api-key": API_KEY}, json=fast, timeout=12)
            if r.status_code == 400:          # model doesn't accept that setting: retry plain
                r = requests.post(url, headers={"x-goog-api-key": API_KEY}, json=body, timeout=12)
            if r.status_code == 200:
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            print("gemini", model, r.status_code, r.text[:200])
            if r.status_code in (429, 503):
                _cool[model] = time.time() + 45
            elif r.status_code in (403, 404):
                _cool[model] = time.time() + 3600
        except Exception as e:
            print("gemini error", model, e)
    return None


def transcribe(pcm):
    import base64
    audio = {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_from_pcm(pcm, IN_RATE)).decode()}}
    body = {"contents": [{"role": "user", "parts": [{"text": TRANSCRIBE}, audio]}], "generationConfig": {"maxOutputTokens": 150}}
    t = _gemini(body)
    print("transcript:", t)
    if not t or "IGNORE" in t:
        return None
    return t.strip()


def listen_and_match(pcm):
    """One Gemini call: transcript + matched row. -> (heard, kind, item) ; heard None = nothing said."""
    import base64
    items = kb.items
    if not items:
        return transcribe(pcm), "NONE", None
    listing = "\n".join(f"{it['id']}: " + " / ".join(it["questions"][:6]) for it in (items if len(items) <= 120 else items[:120]))
    prompt = (f"هذا تسجيل صوتي لطالب يسأل روبوت استعلامات في {kb.university or 'الجامعة'}. "
              "اولا اكتب ما قيل بالضبط نصاً. بعدها اكتب ||| . بعدها اكتب رقم السطر من القائمة ادناه الذي يطابق معنى السؤال "
              "(تجاهل اسم الروبوت اذا نُودي). اذا كان ترحيب او سلام اكتب HELLO، شكر اكتب THANKS، يسأل من انت او شنو تسوي اكتب WHO، "
              "واذا ما يطابق اي سطر بوضوح اكتب NONE.\n"
              "الشكل: <ما قيل> ||| <رقم او كلمة>\nاذا ما في كلام بشري واضح اكتب فقط IGNORE.\n\nالقائمة:\n" + listing)
    audio = {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_from_pcm(pcm, IN_RATE)).decode()}}
    t = _gemini({"contents": [{"role": "user", "parts": [{"text": prompt}, audio]}], "generationConfig": {"maxOutputTokens": 200}})
    print("merged raw:", t)
    if not t or "IGNORE" in t:
        return None, "NONE", None
    if "|||" not in t:
        return t.strip(), None, None            # malformed: caller falls back to the 2-step path
    heard, code = [x.strip() for x in t.split("|||", 1)]
    up = code.upper()
    for k in ("HELLO", "THANKS", "WHO", "NONE"):
        if k in up:
            return heard, k, None
    m = re.search(r"\d+", code)
    if m:
        for it in items:
            if it["id"] == int(m.group()):
                return heard, "item", it
    return heard, "NONE", None


# ---------------------------------------------------------------- wake name
def name_forms():
    forms = set()
    for n in kb.names:
        v = re.sub(r"^يا\s+", "", _norm(n))
        if v:
            forms.add(v)
        if v in ("روبوت",):
            forms.update(LEGACY_NAME_FORMS)
    return forms


def _word_is_name(w, forms):
    if w.startswith("يا") and len(w) > 3:
        w = w[2:]
    for f in forms:
        if " " in f:
            continue
        if w == f or (len(f) >= 4 and len(w) >= 4 and (f in w or difflib.SequenceMatcher(None, w, f).ratio() >= 0.8)):
            return True
    return False


def has_name(heard):
    n = _norm(heard)
    forms = name_forms()
    if any(" " in f and f in n for f in forms):
        return True
    return any(_word_is_name(w, forms) for w in n.split())


def starts_with_name(heard):
    """True only when the utterance BEGINS with the wake name (optionally after a greeting like 'يا' / 'هلا')."""
    n = _norm(heard)
    forms = name_forms()
    words = n.split()
    while words and words[0] in GREETING_WORDS:
        words = words[1:]
    if not words:
        return False
    head = " ".join(words[:3])
    if any(" " in f and head.startswith(f) for f in forms):
        return True
    return _word_is_name(words[0], forms)


def strip_name(heard):
    """Heard text without the wake name and greeting words."""
    n = _norm(heard)
    forms = name_forms()
    for f in forms:
        if " " in f:
            n = n.replace(f, " ")
    words = [w for w in n.split() if not _word_is_name(w, forms) and w not in GREETING_WORDS]
    return " ".join(words)


def only_name(heard):
    return len(strip_name(heard)) < 4


# ---------------------------------------------------------------- matching a question to a sheet row
def _grams(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


def local_score(q, item):
    g = _grams(q)
    best = 0.0
    for nq in item["norm"]:
        h = _grams(nq)
        if g and h:
            best = max(best, len(g & h) / len(g | h))
    return best


def match_question(question):
    """-> ('item', item) | ('HELLO'|'THANKS'|'WHO'|'NONE', None)"""
    items = kb.items
    if not items:
        return "NONE", None
    q = _norm(question)
    cands = items
    if len(items) > 80:
        cands = sorted(items, key=lambda it: -local_score(q, it))[:50]
    listing = "\n".join(f"{it['id']}: " + " / ".join(it["questions"][:6]) for it in cands)
    prompt = (f"انت مصنف اسئلة لروبوت استعلامات في {kb.university or 'الجامعة'}. القائمة ادناه فيها رقم لكل سؤال معروف وصيغه.\n"
              f"{listing}\n\nسؤال الطالب (من تسجيل صوتي وقد فيه اخطاء سمع): «{question}»\n"
              "اكتب فقط رقم السطر الذي يطابق معنى سؤال الطالب. "
              "اذا كان مجرد ترحيب او سلام اكتب HELLO. اذا كان شكر اكتب THANKS. اذا كان يسأل من انت او شنو تسوي اكتب WHO. "
              "اذا ما يطابق اي سطر بوضوح اكتب NONE. لا تكتب اي شرح، فقط الرقم او الكلمة.")
    t = _gemini({"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 100}})
    print("match raw:", t)
    if t:
        up = t.upper()
        for k in ("HELLO", "THANKS", "WHO", "NONE"):
            if k in up:
                return k, None
        m = re.search(r"\d+", t)
        if m:
            for it in items:
                if it["id"] == int(m.group()):
                    return "item", it
        return "NONE", None
    # Gemini unreachable: fall back to local similarity
    best = max(items, key=lambda it: local_score(q, it))
    if local_score(q, best) >= 0.35:
        return "item", best
    return "NONE", None


def canned(kind):
    uni = f" في {kb.university}" if kb.university else ""
    if kind == "HELLO":
        return "هلا والله، شلون أكدر أساعدك؟", "happy"
    if kind == "THANKS":
        return "العفو، بخدمتك دائماً", "love"
    if kind == "WHO":
        return f"أنا روبوت الاستعلامات{uni}. اسألني عن اي معلومة تخص الجامعة.", "happy"
    return "ما عندي هذي المعلومة، راجع مكتب الاستعلامات بالجامعة.", "neutral"


def answer_for(heard, kind=None, item=None):
    """-> (spoken, screen, mood)"""
    if kind is None:
        question = strip_name(heard) or heard
        kind, item = match_question(question)
    if kind == "item":
        return item["spoken"], item["screen"], item["mood"]
    text, mood = canned(kind)
    return text, text, mood


# ---------------------------------------------------------------- voice + screen caches
class LRU:
    def __init__(self, cap):
        self.cap, self.d, self.l = cap, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.l:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
            return v

    def put(self, k, v):
        with self.l:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


tts_cache = LRU(300)
page_cache = LRU(300)


def voice_name():
    v = _norm(kb.voice)
    if v.startswith("رنا") or "rana" in v or "انث" in v:
        return "ar-IQ-RanaNeural"
    if v.startswith("باسل") or "bassel" in v or "ذكر" in v:
        return "ar-IQ-BasselNeural"
    return VOICE


def voice_rate():
    r = (kb.rate or "").strip()
    n = _norm(r)
    if n.startswith("بطي"):
        return "-12%"
    if n.startswith("سريع"):
        return "+10%"
    m = re.fullmatch(r"([+-]?\d{1,2})\s*%?", r)
    if m and r:
        v = max(-40, min(40, int(m.group(1))))
        return f"{v:+d}%"
    return "+0%"


def voice_drive():
    try:
        return max(0.8, min(3.0, float((kb.drive or "").replace(",", "."))))
    except ValueError:
        return VOICE_DRIVE


def apply_pron(text):
    for a, b in kb.prons:
        text = text.replace(a, b)
    return text


async def _tts(text: str, voice: str, rate: str) -> bytes:
    mp3 = b""
    async for part in edge_tts.Communicate(text, voice, rate=rate).stream():
        if part["type"] == "audio":
            mp3 += part["data"]
    return mp3


def speak_pcm(text: str) -> bytes:
    voice, rate, drive = voice_name(), voice_rate(), voice_drive()
    key = f"{voice}|{rate}|{drive}|{text}"
    c = tts_cache.get(key)
    if c is not None:
        return c
    mp3 = asyncio.run(_tts(apply_pron(text), voice, rate))
    dec = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=OUT_RATE)
    samples = array.array("h", dec.samples.tobytes())
    peak = max(max(samples), -min(samples)) if samples else 0
    if peak > 0:                                   # normalise to ~90% of full scale: loud but never clipped
        k = drive                                # soft limiter: louder on average, no harsh clipping
        norm = math.tanh(k)
        samples = array.array("h", (int(30000 * math.tanh(k * v / peak) / norm) for v in samples))
    pcm = samples.tobytes()
    tts_cache.put(key, pcm)
    return pcm


def pages_for(text):
    c = page_cache.get(text)
    if c is None:
        c = arabic_text.render_pages(text)
        page_cache.put(text, c)
    return c


def prewarm():
    """Generate voices + screen pages for all answers in the background so replies are instant."""
    time.sleep(2)
    for t in [greeting_text(), "نعم، اسمعك"] + [canned(k)[0] for k in ("HELLO", "THANKS", "WHO", "NONE")]:
        try:
            speak_pcm(t); pages_for(t)
        except Exception as e:
            print("prewarm error", e)
    for it in list(kb.items):
        try:
            pages_for(it["screen"])
            speak_pcm(it["spoken"])
        except Exception as e:
            print("prewarm error", e)
        time.sleep(0.2)
    print("prewarm done")


def greeting_text():
    if kb.greeting:
        return kb.greeting
    uni = f" في {kb.university}" if kb.university else ""
    return f"أهلاً بيكم{uni}. أنا روبوت الاستعلامات، ناديني {kb.names[0]} واسألني."


def build_body(spoken, screen):
    pages = pages_for(screen)
    gfx = b"".join(pages)
    pcm = speak_pcm(spoken)
    headers = {"X-Gfx": str(len(gfx)), "X-Pages": str(len(pages)), "X-Dur": str(int(len(pcm) / (OUT_RATE * 2) * 1000))}
    return gfx + pcm, headers


# ---------------------------------------------------------------- routes
def authorized() -> bool:
    return (not ROBOT_TOKEN) or request.headers.get("X-Robot-Key") == ROBOT_TOKEN


@app.get("/ping")
def ping():
    return ("ok", 200) if authorized() else ("no", 403)


@app.get("/status")
def status():
    if not authorized():
        return ("no", 403)
    return Response(json.dumps({"university": kb.university, "names": kb.names, "questions": len(kb.items),
                                "source": kb.source, "error": kb.error, "age_s": int(time.time() - kb.loaded_at),
                                "tts_cached": len(tts_cache.d)}, ensure_ascii=False), mimetype="application/json")


@app.get("/reload")
def reload_sheet():
    if not authorized():
        return ("no", 403)
    load_sheet(force=True)
    return status()


@app.get("/hello")
def hello():
    if not authorized():
        return ("no", 403)
    return Response(speak_pcm(greeting_text()), mimetype="application/octet-stream")


@app.post("/talk")
def talk():
    global awake_until
    if not authorized():
        return ("no", 403)
    pcm = request.get_data()
    if len(pcm) < 4000:
        return ("", 204)
    t0 = time.time()
    if time.time() - kb.loaded_at > 20:                # sheet is re-read before answering (the background loop alone was not reliable)
        try:
            load_sheet()
        except Exception as e:
            print("sheet refresh error", e)
    heard, kind, item = listen_and_match(pcm)          # slow network call: not under the lock
    print(f"timing listen+match {time.time() - t0:.1f}s")
    if not heard:
        return ("", 204)
    with lock:
        awake = time.time() < awake_until or request.headers.get("X-Force") == "1"      # X-Force: robot button pressed
        if not awake:
            if not starts_with_name(heard):
                print("ignored:", heard)
                return ("", 204)
            if only_name(heard):
                awake_until = time.time() + AWAKE_WINDOW
                print("wake only ->", heard)
                return Response(speak_pcm("نعم، اسمعك"), mimetype="application/octet-stream", headers={"X-Awake": "1"})
        awake_until = 0
        spoken, screen, mood = answer_for(heard, kind, item)
    print("heard:", heard, "| reply:", spoken, "|", mood)
    try:
        t1 = time.time()
        body, headers = build_body(spoken, screen)
        print(f"timing voice+pages {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s")
    except Exception as e:
        print("tts error", e)
        return ("", 500)
    headers["X-Mood"] = mood
    return Response(body, mimetype="application/octet-stream", headers=headers)


@app.get("/")
def index():
    return "robot server running"


load_sheet(force=True)
threading.Thread(target=sheet_loop, daemon=True).start()
