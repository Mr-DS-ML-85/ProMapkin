"""`promapkin add <url|path>`: bring outside knowledge into the graph.

- arXiv (`https://arxiv.org/abs/1706.03762`, `arXiv:1706.03762`): title, authors and abstract from the arXiv API,
  plus the PDF, whose text is extracted like any repository PDF.
- YouTube: title, channel and description, and a timestamped transcript from the video's captions (yt-dlp), or,
  when there are none, local speech-to-text (faster-whisper). Nothing is uploaded anywhere.
- Any other web page: readable text as Markdown (headings, paragraphs, lists and code blocks kept); PDFs are
  saved as PDFs.
- A local file (PDF, Word, PowerPoint, Markdown, text, image): copied in.

Everything lands in `.promapkin/sources/` with its provenance in `sources/index.json`, then a scan indexes it
as documentation: sections, code mentions, decisions and claims, linked to the code they name. Fetched text is
untrusted data. Only http(s) URLs to public hosts are fetched unless private hosts are explicitly allowed.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

Fetch = Callable[[str], tuple[bytes, str]]   # url -> (body, content type)
MAX_BYTES = 60_000_000


class SourceError(RuntimeError):
    pass


def default_fetch(allow_private: bool = False) -> Fetch:
    def fetch(url: str) -> tuple[bytes, str]:
        check_url(url, allow_private)
        req = urllib.request.Request(url, headers={"User-Agent": "promapkin (+https://github.com/)"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                final = resp.geturl()
                if final != url:
                    check_url(final, allow_private)
                body = resp.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise SourceError(f"{url}: larger than {MAX_BYTES // 1_000_000} MB")
                return body, resp.headers.get("Content-Type", "")
        except OSError as exc:
            raise SourceError(f"cannot fetch {url}: {exc}") from exc
    return fetch


def check_url(url: str, allow_private: bool = False) -> None:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SourceError(f"only http(s) URLs can be added: {url}")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except OSError:
        return   # unresolvable here (e.g. behind a proxy); the fetch itself will fail or succeed via the proxy
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise SourceError(f"{parts.hostname} resolves to a private address ({ip}); pass --allow-private to add "
                              "internal pages")


def classify(target: str) -> tuple[str, str]:
    """(kind, key): arxiv/<id>, youtube/<video id>, web/<url>, file/<path>."""
    m = re.match(r"(?:arxiv:|https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/)([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?/?$",
                 target, re.I)
    if m:
        return "arxiv", m.group(1)
    m = re.match(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
                 r"([\w-]{6,})", target)
    if m:
        return "youtube", m.group(1)
    if re.match(r"https?://", target):
        return "web", target
    return "file", target


def add(ws: Any, target: str, fetch: Fetch | None = None, allow_private: bool = False, title: str | None = None,
        transcribe: Callable[[str, Path], dict[str, Any]] | None = None) -> dict[str, Any]:
    fetch = fetch or default_fetch(allow_private)
    kind, key = classify(target)
    root = ws.dir / "sources"
    root.mkdir(parents=True, exist_ok=True)
    if kind == "arxiv":
        files, meta = _arxiv(root, key, fetch)
    elif kind == "youtube":
        files, meta = _youtube(root, key, target, transcribe or youtube_transcript)
    elif kind == "web":
        files, meta = _web(root, target, fetch)
    else:
        files, meta = _file(root, Path(target).expanduser())
    if title:
        meta["title"] = title
    record = {"kind": kind, "source": target, "title": meta.get("title"), "added_at": time.time(),
              "files": [f.relative_to(ws.root).as_posix() for f in files], **{k: v for k, v in meta.items()
                                                                              if k != "title"}}
    _ledger_add(root, record)
    return record


def list_sources(ws: Any) -> list[dict[str, Any]]:
    path = ws.dir / "sources" / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def remove(ws: Any, ref: str) -> dict[str, Any]:
    root = ws.dir / "sources"
    records = list_sources(ws)
    keep, removed = [], []
    for r in records:
        if ref in (r["source"], r.get("title")) or any(ref in f for f in r["files"]):
            for f in r["files"]:
                (ws.root / f).unlink(missing_ok=True)
            removed.append(r)
        else:
            keep.append(r)
    (root / "index.json").write_text(json.dumps(keep, indent=2) + "\n")
    return {"removed": removed}


def _ledger_add(root: Path, record: dict[str, Any]) -> None:
    path = root / "index.json"
    records = json.loads(path.read_text()) if path.exists() else []
    records = [r for r in records if r["source"] != record["source"]] + [record]
    path.write_text(json.dumps(records, indent=2) + "\n")


def _slug(text: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or hashlib.sha1(text.encode()).hexdigest()[:10]


def _frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items() if v not in (None, "", [])]
    return "\n".join(lines + ["---", ""])


# ------------------------------------------------------------------ arXiv
def _arxiv(root: Path, arxiv_id: str, fetch: Fetch) -> tuple[list[Path], dict[str, Any]]:
    body, _ = fetch(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}")
    xml = body.decode("utf-8", "replace")
    entry = xml[xml.find("<entry>"):] if "<entry>" in xml else ""
    if not entry or "<title>Error</title>" in entry:
        raise SourceError(f"arXiv has no paper {arxiv_id}")

    def tag(name: str) -> str:
        m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", entry, re.S)
        return " ".join(html.unescape(m.group(1)).split()) if m else ""
    meta = {"title": tag("title"), "authors": [" ".join(html.unescape(a).split()) for a in
                                               re.findall(r"<name>(.*?)</name>", entry, re.S)],
            "published": tag("published")[:10], "arxiv_id": arxiv_id, "url": f"https://arxiv.org/abs/{arxiv_id}",
            "categories": re.findall(r'<category term="([^"]+)"', entry)}
    base = f"arxiv-{_slug(arxiv_id, 40)}"
    md = root / f"{base}.md"
    md.write_text(_frontmatter(meta) + f"# {meta['title']}\n\n{', '.join(meta['authors'])} · arXiv {arxiv_id} · "
                  f"{meta['published']}\n\n## Abstract\n\n{tag('summary')}\n")
    files = [md]
    try:
        pdf, ctype = fetch(f"https://arxiv.org/pdf/{arxiv_id}")
        if pdf[:5] == b"%PDF-":
            p = root / f"{base}.pdf"
            p.write_bytes(pdf)
            files.append(p)
    except SourceError as exc:
        meta["pdf_error"] = str(exc)[:200]
    return files, meta


# ------------------------------------------------------------------ YouTube
def youtube_transcript(url: str, workdir: Path) -> dict[str, Any]:
    """Metadata and transcript through yt-dlp (captions first, then local faster-whisper on the audio)."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise SourceError("YouTube needs yt-dlp (pip install yt-dlp); for videos without captions also "
                          "faster-whisper (pip install faster-whisper)")
    info = json.loads(_run([ytdlp, "-J", "--skip-download", "--no-playlist", url]) or "{}")
    _run([ytdlp, "--skip-download", "--no-playlist", "--write-subs", "--write-auto-subs", "--sub-langs",
          "en.*,en,-live_chat", "--sub-format", "vtt", "-o", str(workdir / "%(id)s.%(ext)s"), url])
    vtts = sorted(workdir.glob("*.vtt"), key=lambda p: (".en." not in p.name, len(p.name)))
    segments: list[tuple[float, str]] = []
    method = "captions"
    if vtts:
        segments = parse_vtt(vtts[0].read_text(errors="replace"))
    else:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise SourceError("this video has no captions; install faster-whisper to transcribe it locally") from exc
        _run([ytdlp, "-x", "--audio-format", "m4a", "--no-playlist", "-o", str(workdir / "audio.%(ext)s"), url],
             timeout=1800)
        audio = next(workdir.glob("audio.*"), None)
        if audio is None:
            raise SourceError("could not download the audio track")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments = [(seg.start, seg.text.strip()) for seg in model.transcribe(str(audio))[0]]
        method = "faster-whisper (local)"
    return {"title": info.get("title"), "channel": info.get("channel") or info.get("uploader"),
            "duration_seconds": info.get("duration"), "upload_date": info.get("upload_date"),
            "description": (info.get("description") or "")[:4000], "segments": segments, "transcription": method}


def _run(cmd: list[str], timeout: int = 300) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceError(f"{Path(cmd[0]).name} failed: {exc}") from exc
    if r.returncode != 0:
        raise SourceError(f"{Path(cmd[0]).name} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def parse_vtt(text: str) -> list[tuple[float, str]]:
    """WebVTT cues -> [(start seconds, text)], removing the rolling duplicates of auto-generated captions."""
    out: list[tuple[float, str]] = []
    seen_tail = ""
    for block in re.split(r"\n\s*\n", text):
        m = re.search(r"(\d+):(\d\d):(\d\d)[.,](\d+)\s+-->", block) or re.search(r"(\d\d):(\d\d)[.,](\d+)\s+-->", block)
        if not m:
            continue
        g = m.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) if len(g) == 4 else int(g[0]) * 60 + int(g[1])
        lines = [re.sub(r"<[^>]+>", "", ln).strip() for ln in block.split("\n")[1:] if "-->" not in ln]
        line = " ".join(ln for ln in lines if ln)
        if not line or line == seen_tail or seen_tail.endswith(line):
            continue
        if line.startswith(seen_tail) and seen_tail:
            line = line[len(seen_tail):].strip()
        seen_tail = " ".join(lines[-1:]) or line
        if line:
            out.append((float(start), html.unescape(line)))
    return out


def _youtube(root: Path, video_id: str, url: str, transcribe: Callable[[str, Path], dict[str, Any]]
             ) -> tuple[list[Path], dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        data = transcribe(url, Path(tmp))
    meta = {"title": data.get("title") or f"YouTube {video_id}", "channel": data.get("channel"),
            "url": f"https://www.youtube.com/watch?v={video_id}", "duration_seconds": data.get("duration_seconds"),
            "transcription": data.get("transcription")}
    chunks, cur, cur_start = [], [], None
    for start, text in data.get("segments", []):
        if cur_start is None:
            cur_start = start
        cur.append(text)
        if start - cur_start >= 120:
            chunks.append((cur_start, " ".join(cur)))
            cur, cur_start = [], None
    if cur:
        chunks.append((cur_start or 0.0, " ".join(cur)))
    body = [f"# {meta['title']}", "", f"{meta.get('channel') or ''} · {meta['url']}", ""]
    if data.get("description"):
        body += ["## Description", "", data["description"], ""]
    body += ["## Transcript", ""]
    for start, text in chunks:
        body += [f"### [{int(start) // 60:02d}:{int(start) % 60:02d}]", "", text, ""]
    md = root / f"youtube-{_slug(video_id, 20)}.md"
    md.write_text(_frontmatter(meta) + "\n".join(body))
    meta["segments"] = len(data.get("segments", []))
    return [md], meta


# ------------------------------------------------------------------ web pages
def _web(root: Path, url: str, fetch: Fetch) -> tuple[list[Path], dict[str, Any]]:
    body, ctype = fetch(url)
    slug = _slug(re.sub(r"^https?://(www\.)?", "", url), 70)
    if body[:5] == b"%PDF-" or "pdf" in ctype:
        p = root / f"web-{slug}.pdf"
        p.write_bytes(body)
        return [p], {"title": Path(urllib.parse.urlparse(url).path).name or url, "url": url}
    text = body.decode(_charset(ctype) or "utf-8", "replace")
    if "html" not in ctype and not re.search(r"<html|<body|<p[ >]", text[:5000], re.I):
        md_text, title = text, url
    else:
        md_text, title = html_to_markdown(text)
    meta = {"title": title or url, "url": url}
    p = root / f"web-{slug}.md"
    p.write_text(_frontmatter(meta) + (md_text if md_text.lstrip().startswith("#") else f"# {meta['title']}\n\n"
                                       + md_text))
    return [p], meta


def _charset(ctype: str) -> str | None:
    m = re.search(r"charset=([\w-]+)", ctype or "")
    return m.group(1) if m else None


def html_to_markdown(page: str) -> tuple[str, str]:
    """Readable text of an HTML page: the <main>/<article> when present, headings, paragraphs, lists, code."""
    title_m = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    title = " ".join(html.unescape(title_m.group(1)).split()) if title_m else ""
    body = page
    for tag in ("main", "article"):
        m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", page, re.S | re.I)
        if m and len(m.group(1)) > 500:
            body = m.group(1)
            break
    body = re.sub(r"<(script|style|nav|header|footer|aside|noscript|svg|form)\b.*?</\1>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    codes: list[str] = []

    def keep_code(m: re.Match[str]) -> str:
        codes.append(html.unescape(re.sub(r"<[^>]+>", "", m.group(1))))
        return f"\n\x00CODE{len(codes) - 1}\x00\n"
    body = re.sub(r"<pre\b[^>]*>(.*?)</pre>", keep_code, body, flags=re.S | re.I)
    for level in range(1, 5):
        body = re.sub(rf"<h{level}\b[^>]*>(.*?)</h{level}>",
                      lambda m, lv=level: f"\n\n{'#' * lv} {' '.join(re.sub(r'<[^>]+>', '', m.group(1)).split())}\n\n",
                      body, flags=re.S | re.I)
    body = re.sub(r"<li\b[^>]*>", "\n- ", body, flags=re.I)
    body = re.sub(r"<code\b[^>]*>(.*?)</code>", lambda m: f"`{re.sub(r'<[^>]+>', '', m.group(1))}`", body,
                  flags=re.S | re.I)
    body = re.sub(r"</?(p|div|section|br|tr|table|ul|ol|blockquote)\b[^>]*>", "\n", body, flags=re.I)
    body = html.unescape(re.sub(r"<[^>]+>", "", body))
    lines = [" ".join(ln.split()) for ln in body.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    text = re.sub(r"\x00CODE(\d+)\x00", lambda m: f"```\n{codes[int(m.group(1))].strip()}\n```", text)
    return text, title


# ------------------------------------------------------------------ local files
def _file(root: Path, path: Path) -> tuple[list[Path], dict[str, Any]]:
    if not path.is_file():
        raise SourceError(f"not a file or URL: {path}")
    dest = root / f"file-{_slug(path.stem, 50)}{path.suffix.lower()}"
    shutil.copyfile(path, dest)
    return [dest], {"title": path.stem, "original_path": str(path)}
