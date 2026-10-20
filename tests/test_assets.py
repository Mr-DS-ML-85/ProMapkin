"""Assets: PDFs, Office documents, images and video, built here byte by byte so no binary fixtures are needed."""

import io
import json
import struct
import unittest
import zipfile
import zlib

from tests.helpers import RepoTestCase, git
from promapkin.extract import assets_ext


def make_pdf(lines: list[str], title: str = "Search design") -> bytes:
    content = b"BT /F1 12 Tf 72 720 Td " + b" T* ".join(
        b"(" + line.encode("latin-1").replace(b"(", b"\\(").replace(b")", b"\\)") + b") Tj" for line in lines) + b" ET"
    stream = zlib.compress(content)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 "
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n" + stream +
            b"\nendstream",
            b"<< /Title (" + title.encode() + b") /Author (Docs Team) >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    return out + (f"trailer << /Size {len(objs) + 1} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{xref}\n"
                  f"%%EOF\n").encode()


def make_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document><w:body>'
                   '<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Ranking design</w:t></w:r></w:p>'
                   '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Scoring</w:t></w:r></w:p>'
                   '<w:p><w:r><w:t xml:space="preserve">The </w:t></w:r><w:r><w:t>SearchService</w:t></w:r>'
                   '<w:r><w:t> scores candidates &amp; ties.</w:t></w:r></w:p></w:body></w:document>')
        z.writestr("docProps/core.xml", "<cp:coreProperties><dc:title>Ranking design</dc:title></cp:coreProperties>")
    return buf.getvalue()


def make_pptx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, (head, body) in enumerate([("Overview", "SearchService coordinates ranking"),
                                          ("Storage", "IndexRepository persists documents")], 1):
            z.writestr(f"ppt/slides/slide{i}.xml", f"<p:sld><a:p><a:r><a:t>{head}</a:t></a:r></a:p>"
                                                   f"<a:p><a:r><a:t>{body}</a:t></a:r></a:p></p:sld>")
    return buf.getvalue()


def make_png(w: int = 3, h: int = 2) -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def make_jpeg(w: int = 640, h: int = 480) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def make_mp4(seconds: float = 5.0) -> bytes:
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"\x00\x00\x02\x00"
    mvhd_body = b"\x00\x00\x00\x00" + struct.pack(">IIII", 0, 0, 1000, int(seconds * 1000)) + b"\x00" * 80
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    return ftyp + moov


SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><title>Search architecture</title>'
       '<desc>Request path</desc><text x="10" y="20">SearchService</text><text x="200" y="20">IndexRepository</text>'
       '</svg>')


class AssetUnitTest(unittest.TestCase):
    def test_pdf_builtin_text_and_metadata(self):
        meta, text = assets_ext.pdf_text(make_pdf(["SearchService uses IndexRepository (sqlite).", "Second line"]))
        self.assertEqual(meta["pages"], 1)
        self.assertEqual(meta["title"], "Search design")
        self.assertIn("SearchService uses IndexRepository (sqlite).", text)
        self.assertIn("Second line", text)

    def test_office(self):
        text, title = assets_ext.office_text(make_docx(), ".docx")
        self.assertEqual(title, "Ranking design")
        self.assertIn("## Scoring", text)
        self.assertIn("The SearchService scores candidates & ties.", text)
        text, _ = assets_ext.office_text(make_pptx(), ".pptx")
        self.assertIn("## Slide 2: Storage", text)

    def test_image_and_video_metadata(self):
        self.assertEqual(assets_ext.image_info(make_png(3, 2), ".png"), {"width": 3, "height": 2})
        self.assertEqual(assets_ext.image_info(make_jpeg(640, 480), ".jpg"), {"width": 640, "height": 480})
        self.assertEqual(assets_ext.image_info(SVG.encode(), ".svg"), {"width": 400.0, "height": 200.0})
        text, title = assets_ext.svg_text(SVG.encode())
        self.assertEqual(title, "Search architecture")
        self.assertIn("IndexRepository", text)
        self.assertEqual(assets_ext.video_info(make_mp4(5.0), ".mp4")["duration_seconds"], 5.0)


class AssetScanTest(RepoTestCase):
    scan_on_setup = False

    def setUp(self):
        super().setUp()
        d = self.root / "docs" / "assets"
        d.mkdir(parents=True)
        (d / "design.pdf").write_bytes(make_pdf(["The SearchService calls IndexRepository for reads."]))
        (d / "ranking.docx").write_bytes(make_docx())
        (d / "overview.pptx").write_bytes(make_pptx())
        (d / "arch.svg").write_text(SVG)
        (d / "screenshot.png").write_bytes(make_png())
        (d / "demo.mp4").write_bytes(make_mp4(12.5))
        doc = self.root / "docs" / "architecture.md"
        doc.write_text(doc.read_text() + "\n\n![Architecture](assets/arch.svg)\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "assets")

    def test_assets_are_indexed_searchable_and_linked(self):
        from promapkin.scanner import scan
        from promapkin.service import Service
        r = scan(self.ws)
        self.assertEqual(r["errors"], [])
        svc = Service(self.ws)
        assets = {a["file"].rsplit("/", 1)[-1]: a for a in svc.assets()["assets"]}
        self.assertEqual(set(assets), {"design.pdf", "ranking.docx", "overview.pptx", "arch.svg", "screenshot.png",
                                       "demo.mp4"})
        self.assertEqual(assets["design.pdf"]["pages"], 1)
        self.assertTrue(assets["design.pdf"]["has_text"])
        self.assertEqual(assets["demo.mp4"]["duration_seconds"], 12.5)
        self.assertEqual((assets["screenshot.png"]["width"], assets["screenshot.png"]["height"]), (3, 2))
        self.assertTrue(any(r.startswith("doc:docs/architecture.md") for r in assets["arch.svg"]["referenced_by"]))
        # Text from binary documents is searchable and linked to the code it names.
        hits = [h["id"] for h in svc.search("SearchService calls IndexRepository", limit=10)["results"]]
        self.assertTrue(any(h.startswith("doc:docs/assets/design.pdf") for h in hits), hits)
        self.assertTrue(any(m.endswith("SearchService") for m in assets["design.pdf"]["mentions_code"]))
        self.assertTrue(any(m.endswith("SearchService") for m in assets["ranking.docx"]["mentions_code"]))

    def test_llm_vision_describes_diagrams(self):
        from promapkin.llm.engine import LLMEngine
        from promapkin.scanner import scan
        from tests.test_robustness import ScriptedClient

        class Vision(ScriptedClient):
            def json(self, system, user, schema, *, max_tokens=16000, effort="medium", name="result", images=None):
                if name == "describe_image":
                    self.calls.append({"task": name, "images": images, "locked": self.ws.store.conn.in_transaction})
                    return {"kind": "architecture_diagram", "summary": "Request flow of the search service",
                            "components": ["SearchService", "IndexRepository"], "text_in_image": "",
                            "relationships": [{"source": "SearchService", "target": "IndexRepository",
                                               "label": "reads from"}]}
                return super().json(system, user, schema, max_tokens=max_tokens, effort=effort, name=name)
        client = Vision(self.ws)
        self.ws.llm = LLMEngine(self.ws, client)
        scan(self.ws)
        calls = [c for c in client.calls if c["task"] == "describe_image"]
        self.assertEqual(len(calls), 1)                       # the PNG; SVG text is read directly
        self.assertEqual(calls[0]["images"][0][0], "image/png")
        self.assertFalse(calls[0]["locked"])
        node = self.ws.store.node("asset:docs/assets/screenshot.png")
        self.assertEqual(node["attrs"]["image_kind"], "architecture_diagram")
        claims = [e for e in self.ws.store.edges(types=["reads"]) if e["attrs"].get("claim")]
        self.assertTrue(any(e["src"].endswith("SearchService") and e["dst"].endswith("IndexRepository")
                            for e in claims), json.dumps(claims, default=str)[:400])


if __name__ == "__main__":
    unittest.main()
