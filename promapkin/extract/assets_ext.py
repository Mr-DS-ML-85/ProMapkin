"""Non-code assets: PDFs, Office documents, images and video.

What is extracted, without dependencies:
- PDF: metadata (title, author, pages) and text from uncompressed and Flate-compressed content streams. With
  `pypdf` installed (pip install 'promapkin[assets]') its text extraction is used instead (handles font encodings).
- Word (.docx) and PowerPoint (.pptx): paragraphs, headings and slide text (they are zip archives of XML).
- Images: format and dimensions (PNG, JPEG, GIF, WebP, BMP); SVG titles, descriptions and text labels.
  With `pytesseract` and the tesseract binary installed, raster images are OCR'd. In --llm mode a vision model
  describes diagrams and screenshots (components and the relationships drawn between them).
- Video: container, size and (MP4/MOV) duration. Video content itself is not analysed.

Extracted text goes through the documentation extractor, so a PDF or a slide deck gets the same sections,
code mentions, decisions and claims as a Markdown file, linked to the code it names.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
from pathlib import PurePosixPath
from typing import Any

from promapkin import evidence as ev
from promapkin.extract import Facts

ASSET_LANGUAGES = {"pdf", "office", "image", "video"}
ASSET_EXTENSIONS = {".pdf": "pdf", ".docx": "office", ".pptx": "office", ".png": "image", ".jpg": "image",
                    ".jpeg": "image", ".gif": "image", ".webp": "image", ".bmp": "image", ".svg": "image",
                    ".mp4": "video", ".mov": "video", ".m4v": "video", ".webm": "video", ".mkv": "video",
                    ".avi": "video"}
MAX_ASSET_BYTES = {"pdf": 40_000_000, "office": 40_000_000, "image": 15_000_000, "video": 0}   # 0: never read
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml"}


def extract(path: str, data: bytes, language: str, llm: Any = None, size: int | None = None) -> Facts:
    suffix = PurePosixPath(path).suffix.lower()
    size = size if size is not None else len(data)
    attrs: dict[str, Any] = {"media": language, "format": suffix.lstrip("."), "bytes": size}
    text, title, errors = "", None, []
    try:
        if language == "pdf":
            meta, text = pdf_text(data)
            attrs.update(meta)
            title = meta.get("title")
        elif language == "office":
            text, title = office_text(data, suffix)
        elif language == "image":
            attrs.update(image_info(data, suffix))
            if suffix == ".svg":
                text, title = svg_text(data)
            else:
                text = ocr(data)
                if text:
                    attrs["ocr"] = True
        elif language == "video":
            attrs.update(video_info(data, suffix))
    except (ValueError, KeyError, IndexError, struct.error, zipfile.BadZipFile, zlib.error) as exc:
        errors.append(f"{language} extraction incomplete: {type(exc).__name__}: {exc}")
    description = None
    if llm is not None and language == "image" and suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp") \
            and 0 < size <= 5_000_000:
        try:
            description = llm.describe_image(path, data, MEDIA_TYPES[suffix])
        except Exception as exc:  # noqa: BLE001 - optional; the asset is still indexed
            errors.append(f"LLM image description failed ({str(exc)[:160]})")
    return _facts(path, language, attrs, text, title, description, errors)


def _facts(path: str, language: str, attrs: dict[str, Any], text: str, title: str | None,
           description: dict[str, Any] | None, errors: list[str]) -> Facts:
    from promapkin.extract.docs_ext import extract as docs_extract

    name = PurePosixPath(path).name
    body = text.strip()
    if description:
        body = (body + "\n\n" if body else "") + _description_markdown(description)
    if body:
        md = body if body.lstrip().startswith("#") else f"# {title or name}\n\n{body}"
        facts = docs_extract(path, md[:400_000])
        facts.language = language
        doc = next(n for n in facts.nodes if n["kind"] == "document")
        doc["attrs"].update({**attrs, "extracted_text": True})
        doc["evidence"] = ev.DOCUMENTATION
        for n in facts.nodes:          # text lines of a binary file have no stable line numbers
            if n["kind"] == "doc_section":
                n["attrs"] = {**n.get("attrs", {}), "derived_from": language}
    else:
        facts = Facts(path, language)
    aid = f"asset:{path}"
    facts.node(aid, "asset", title or name, ev.MANIFEST, file=path, text=(text or "")[:600] or None,
               attrs={**attrs, "has_text": bool(text.strip()), "described": bool(description)})
    facts.edge(f"file:{path}", aid, "defines", ev.MANIFEST)
    if body:
        facts.edge(aid, f"doc:{path}", "contains", ev.DOCUMENTATION)
    if description:
        facts.nodes[-1]["attrs"]["description"] = description.get("summary")
        facts.nodes[-1]["attrs"]["image_kind"] = description.get("kind")
        for rel in description.get("relationships", [])[:40]:
            if rel.get("source") and rel.get("target"):
                facts.ref(f"doc:{path}", rel["target"], "claims", None, subject=rel["source"],
                          verb=(rel.get("label") or "depends on").lower()[:40],
                          sentence=f"diagram: {rel['source']} -> {rel['target']} ({rel.get('label', '')})"[:200],
                          extractor="llm-vision")
    facts.errors += errors
    return facts


def _description_markdown(d: dict[str, Any]) -> str:
    lines = [f"## Image description ({d.get('kind', 'image')})", "", d.get("summary", "")]
    if d.get("components"):
        lines += ["", "Components: " + ", ".join(f"`{c}`" for c in d["components"][:40])]
    if d.get("text_in_image"):
        lines += ["", "Text in image: " + d["text_in_image"][:2000]]
    return "\n".join(lines)


# ------------------------------------------------------------------ PDF
def pdf_text(data: bytes) -> tuple[dict[str, Any], str]:
    meta: dict[str, Any] = {"pages": len(re.findall(rb"/Type\s*/Page(?!s)\b", data))}
    for key in ("Title", "Author", "Subject"):
        m = re.search(rb"/" + key.encode() + rb"\s*\((.*?)(?<!\\)\)", data, re.S)
        if m:
            meta[key.lower()] = _pdf_string(m.group(1))[:300]
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        PdfReader = None
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages[:300]]
            if any(t.strip() for t in pages):
                meta.update({"pages": len(reader.pages), "text_extractor": "pypdf"})
                return meta, "\n\n".join(f"## Page {i + 1}\n\n{t.strip()}" for i, t in enumerate(pages)
                                           if t.strip())
        except Exception:  # noqa: BLE001 - fall back to the built-in reader
            pass
    meta["text_extractor"] = "built-in"
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = m.group(1)
        header = data[max(0, m.start() - 300):m.start()]
        if b"/FlateDecode" in header:
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                continue
        elif b"/Filter" in header:
            continue                       # images and other encodings
        if b"BT" not in raw:
            continue
        text = _pdf_content_text(raw)
        if text.strip():
            chunks.append(text)
    text = "\n\n".join(chunks)
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    if text and printable / len(text) < 0.85:
        meta["text_note"] = "fonts use custom encodings; install pypdf for text"
        return meta, ""
    return meta, text


def _pdf_content_text(stream: bytes) -> str:
    out, line = [], []
    for m in re.finditer(rb"\((?:\\.|[^\\)])*\)\s*Tj|\[(.*?)\]\s*TJ|T\*|\bTd\b|\bTD\b|\bET\b|'|\"", stream, re.S):
        tok = m.group(0)
        if tok.endswith(b"Tj"):
            line.append(_pdf_string(tok[1:tok.rindex(b")")]))
        elif tok.endswith(b"TJ"):
            for part in re.finditer(rb"\(((?:\\.|[^\\)])*)\)|(-?\d+(?:\.\d+)?)", m.group(1) or b""):
                if part.group(1) is not None:
                    line.append(_pdf_string(part.group(1)))
                elif part.group(2) and float(part.group(2)) < -200:
                    line.append(" ")
        else:
            if line:
                out.append("".join(line))
                line = []
    if line:
        out.append("".join(line))
    return "\n".join(s for s in out if s.strip())


def _pdf_string(raw: bytes) -> str:
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    s = re.sub(rb"\\([nrtbf()\\])", lambda m: {b"n": b"\n", b"r": b"", b"t": b"\t", b"b": b"", b"f": b"",
                                               b"(": b"(", b")": b")", b"\\": b"\\"}[m.group(1)], raw)
    s = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8) & 0xFF]), s)
    return s.decode("latin-1")


# ------------------------------------------------------------------ Office
def office_text(data: bytes, suffix: str) -> tuple[str, str | None]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        title = None
        if "docProps/core.xml" in z.namelist():
            m = re.search(rb"<dc:title>(.*?)</dc:title>", z.read("docProps/core.xml"), re.S)
            title = _xml_text(m.group(1)) if m else None
        if suffix == ".docx":
            xml = z.read("word/document.xml")
            out = []
            for p in re.findall(rb"<w:p[ >].*?</w:p>", xml, re.S):
                text = "".join(_xml_text(t) for t in re.findall(rb"<w:t[^>]*>(.*?)</w:t>", p, re.S))
                if not text.strip():
                    continue
                style = re.search(rb'<w:pStyle w:val="(?:Heading|heading)\s?(\d)"', p)
                title_style = re.search(rb'<w:pStyle w:val="Title"', p)
                if title_style:
                    out.append(f"# {text}")
                    title = title or text
                elif style:
                    out.append(f"{'#' * min(6, int(style.group(1)) + 1)} {text}")
                else:
                    out.append(text)
            return "\n\n".join(out), title
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group(0)))
        out = []
        for i, name in enumerate(slides, 1):
            xml = z.read(name)
            paras = ["".join(_xml_text(t) for t in re.findall(rb"<a:t>(.*?)</a:t>", p, re.S))
                     for p in re.findall(rb"<a:p>.*?</a:p>", xml, re.S)]
            paras = [p for p in paras if p.strip()]
            if paras:
                out.append(f"## Slide {i}: {paras[0][:80]}\n\n" + "\n\n".join(paras[1:]))
        return "\n\n".join(out), title


def _xml_text(raw: bytes) -> str:
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<[^>]+>", "", s)
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
            .replace("&amp;", "&"))


# ------------------------------------------------------------------ images
def image_info(data: bytes, suffix: str) -> dict[str, Any]:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return {"width": w, "height": h}
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return {"width": w, "height": h}
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return {"width": w, "height": abs(h)}
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return {"width": w, "height": h}
        if chunk == b"VP8L":
            b = data[21:25]
            w = 1 + (((b[1] & 0x3F) << 8) | b[0])
            h = 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
            return {"width": w, "height": h}
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return {"width": w & 0x3FFF, "height": h & 0x3FFF}
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return {"width": w, "height": h}
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg
    if suffix == ".svg":
        head = data[:2000].decode("utf-8", "replace")
        w = re.search(r'\bwidth="([\d.]+)', head)
        h = re.search(r'\bheight="([\d.]+)', head)
        vb = re.search(r'viewBox="[\d.\s-]+?\s([\d.]+)\s([\d.]+)"', head)
        if w and h:
            return {"width": float(w.group(1)), "height": float(h.group(1))}
        if vb:
            return {"width": float(vb.group(1)), "height": float(vb.group(2))}
    return {}


def svg_text(data: bytes) -> tuple[str, str | None]:
    xml = data.decode("utf-8", "replace")
    title = re.search(r"<title[^>]*>(.*?)</title>", xml, re.S)
    desc = re.findall(r"<desc[^>]*>(.*?)</desc>", xml, re.S)
    labels = [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r"<text[^>]*>(.*?)</text>", xml, re.S)]
    labels += [m.strip() for m in re.findall(r'aria-label="([^"]+)"', xml)]
    parts = [d.strip() for d in desc if d.strip()] + [label for label in labels if label]
    out = "\n".join(dict.fromkeys(parts))
    return (_xml_text(out.encode()) if out else ""), (_xml_text(title.group(1).encode()).strip() if title else None)


def ocr(data: bytes) -> str:
    """OCR when pytesseract, Pillow and the tesseract binary are available; otherwise ''."""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)), timeout=20)[:20000]
    except Exception:  # noqa: BLE001 - binary missing, unsupported image, timeout
        return ""


# ------------------------------------------------------------------ video
def video_info(head: bytes, suffix: str) -> dict[str, Any]:
    """Container and (MP4/MOV) duration from the first bytes of the file."""
    info: dict[str, Any] = {"container": suffix.lstrip(".")}
    m = head.find(b"mvhd")
    if m > 0 and len(head) > m + 24:
        version = head[m + 4]
        if version == 1 and len(head) > m + 36:
            timescale, duration = struct.unpack(">IQ", head[m + 24:m + 36])
        else:
            timescale, duration = struct.unpack(">II", head[m + 16:m + 24])
        if timescale:
            info["duration_seconds"] = round(duration / timescale, 2)
    return info
