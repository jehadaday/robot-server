"""Render Arabic text into 240x240 1-bit pages for the robot's screen.
No external Arabic libraries: own joining (presentation forms) + simple RTL line layout."""
import os
import re

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "DejaVuSans-Bold.ttf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
W = H = 240
FONT_SIZE = 27
LINE_H = 37
MARGIN = 8
TEXT_TOP = 14
MAX_LINES = 5
MAX_PAGES = 6

# letter -> (isolated, final, initial, medial); initial/medial None = right-joining only
F = {
    0x0621: (0xFE80, None, None, None),
    0x0622: (0xFE81, 0xFE82, None, None), 0x0623: (0xFE83, 0xFE84, None, None),
    0x0624: (0xFE85, 0xFE86, None, None), 0x0625: (0xFE87, 0xFE88, None, None),
    0x0626: (0xFE89, 0xFE8A, 0xFE8B, 0xFE8C), 0x0627: (0xFE8D, 0xFE8E, None, None),
    0x0628: (0xFE8F, 0xFE90, 0xFE91, 0xFE92), 0x0629: (0xFE93, 0xFE94, None, None),
    0x062A: (0xFE95, 0xFE96, 0xFE97, 0xFE98), 0x062B: (0xFE99, 0xFE9A, 0xFE9B, 0xFE9C),
    0x062C: (0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0), 0x062D: (0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4),
    0x062E: (0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8), 0x062F: (0xFEA9, 0xFEAA, None, None),
    0x0630: (0xFEAB, 0xFEAC, None, None), 0x0631: (0xFEAD, 0xFEAE, None, None),
    0x0632: (0xFEAF, 0xFEB0, None, None), 0x0633: (0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4),
    0x0634: (0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8), 0x0635: (0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC),
    0x0636: (0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0), 0x0637: (0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4),
    0x0638: (0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8), 0x0639: (0xFEC9, 0xFECA, 0xFECB, 0xFECC),
    0x063A: (0xFECD, 0xFECE, 0xFECF, 0xFED0), 0x0641: (0xFED1, 0xFED2, 0xFED3, 0xFED4),
    0x0642: (0xFED5, 0xFED6, 0xFED7, 0xFED8), 0x0643: (0xFED9, 0xFEDA, 0xFEDB, 0xFEDC),
    0x0644: (0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0), 0x0645: (0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4),
    0x0646: (0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8), 0x0647: (0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC),
    0x0648: (0xFEED, 0xFEEE, None, None), 0x0649: (0xFEEF, 0xFEF0, None, None),
    0x064A: (0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4),
    0x067E: (0xFB56, 0xFB57, 0xFB58, 0xFB59), 0x0686: (0xFB7A, 0xFB7B, 0xFB7C, 0xFB7D),
    0x0698: (0xFB8A, 0xFB8B, None, None), 0x06AF: (0xFB92, 0xFB93, 0xFB94, 0xFB95),
    0x06A4: (0xFB6A, 0xFB6B, 0xFB6C, 0xFB6D),
}
LAM_ALEF = {0x0622: (0xFEF5, 0xFEF6), 0x0623: (0xFEF7, 0xFEF8), 0x0625: (0xFEF9, 0xFEFA), 0x0627: (0xFEFB, 0xFEFC)}
TASHKEEL = re.compile(r"[ً-ٰٟـ]")
DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_font = None


def font():
    global _font
    if _font is None:
        for p in FONT_PATHS:
            if os.path.exists(p):
                _font = ImageFont.truetype(p, FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
                break
        else:
            raise RuntimeError("no font")
    return _font


def _is_ar(ch):
    return ord(ch) in F


def shape(word):
    """Logical Arabic string -> list of presentation-form chars (still in logical order)."""
    s = TASHKEEL.sub("", word).translate(DIGITS)
    out, n, i = [], len(s), 0
    prev_joins = False            # previous char connects to the left (towards the next letter)
    while i < n:
        c = ord(s[i])
        if c not in F:
            out.append(s[i]); prev_joins = False; i += 1
            continue
        # lam + alef ligature
        if c == 0x0644 and i + 1 < n and ord(s[i + 1]) in LAM_ALEF:
            iso, fin = LAM_ALEF[ord(s[i + 1])]
            out.append(chr(fin if prev_joins else iso))
            prev_joins = False
            i += 2
            continue
        iso, fin, ini, med = F[c]
        nxt = ord(s[i + 1]) if i + 1 < n else None
        next_conn = nxt is not None and nxt in F and F[nxt][1] is not None
        dual = ini is not None
        if prev_joins and dual and next_conn: g = med
        elif prev_joins and fin is not None: g = fin
        elif dual and next_conn: g = ini
        else: g = iso
        out.append(chr(g))
        prev_joins = dual and next_conn
        i += 1
    return out


def _visual(words):
    """Words in logical (reading) order of an RTL line -> string in left-to-right drawing order."""
    parts = []
    for w in words:
        if any(_is_ar(ch) for ch in w):
            # split into Arabic part and number/latin runs so numbers keep their order
            toks = re.findall(r"[\d][\d,.٫٬]*|[A-Za-z][A-Za-z0-9._%-]*|[^\dA-Za-z]+", TASHKEEL.sub("", w).translate(DIGITS))
            vis = []
            for t in toks:
                if re.match(r"[\dA-Za-z]", t):
                    vis.append(t)
                else:
                    vis.append("".join(reversed(shape(t))))
            parts.append("".join(reversed(vis)))
        else:
            parts.append(w.translate(DIGITS))      # pure number / latin word: keep as is
    return " ".join(reversed(parts))


def _width(words):
    return font().getlength(_visual(words))


def wrap(text, max_w):
    lines, cur = [], []
    for w in text.split():
        if cur and _width(cur + [w]) > max_w:
            lines.append(cur); cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _pack(img):
    img = img.convert("1")
    px = img.load()
    out = bytearray()
    for y in range(H):
        for bx in range(W // 8):
            v = 0
            for b in range(8):
                if px[bx * 8 + b, y]:
                    v |= 0x80 >> b
            out.append(v)
    return bytes(out)


def render_pages(text):
    """-> list of bytes, each 30*240 = 7200 bytes (1bpp, MSB first, 1 = lit)."""
    text = re.sub(r"\s+", " ", text or "").strip() or "-"
    lines = wrap(text, W - 2 * MARGIN)
    groups = [lines[i:i + MAX_LINES] for i in range(0, len(lines), MAX_LINES)][:MAX_PAGES]
    # balance: avoid a last page with a single short orphan line when possible
    pages = []
    f = font()
    for pi, grp in enumerate(groups):
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        block_h = len(grp) * LINE_H
        area_h = 196
        y = TEXT_TOP + max(0, (area_h - block_h) // 2)
        for words in grp:
            vis = _visual(words)
            w = f.getlength(vis)
            d.text((W - MARGIN - w, y), vis, font=f, fill=255)     # right-aligned (RTL)
            y += LINE_H
        if len(groups) > 1:
            n = len(groups)
            step = 16
            x0 = W // 2 - (n - 1) * step // 2
            for k in range(n):
                r = 5 if k == pi else 3
                cx, cy = x0 + k * step, 226
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255 if k == pi else 0, outline=255)
        pages.append(_pack(img))
    return pages


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "أجور قسم هندسة الكهرباء هي 1,500,000 دينار عراقي سنوياً"
    for i, p in enumerate(render_pages(t)):
        im = Image.new("1", (W, H))
        px = im.load()
        for y in range(H):
            for x in range(W):
                px[x, y] = 1 if p[y * 30 + x // 8] & (0x80 >> (x % 8)) else 0
        im.convert("L").save(f"/tmp/claude-0/page{i}.png")
        print("page", i)
