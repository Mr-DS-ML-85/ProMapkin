"""Infrastructure and schema awareness: the sample app plus Dockerfile, Compose, Kubernetes, Terraform, CI,
OpenAPI, SQL, Prisma, GraphQL and protobuf files (tests/fixtures/infra_overlay)."""

import shutil
import unittest
from pathlib import Path

from tests.helpers import RepoTestCase, git
from promapkin.extract.yamlish import load_all
from promapkin.extract.infra_ext import command_targets, hcl_blocks
from promapkin.extract.schema_ext import sql_statements, tables_in

OVERLAY = Path(__file__).parent / "fixtures" / "infra_overlay"


class ParsersTest(unittest.TestCase):
    def test_yaml_subset(self):
        docs = load_all("a: 1\nb:\n  - x: 1\n    y: [1, 2]\n  - plain\nc: |\n  line1\n  line2\n---\n"
                        "d: {k: v}  # comment\n{{- if .Values.x }}\ne: 'q'\n")
        self.assertEqual(docs[0], {"a": 1, "b": [{"x": 1, "y": [1, 2]}, "plain"], "c": "line1\nline2"})
        self.assertEqual(docs[1], {"d": {"k": "v"}, "e": "q"})
        self.assertEqual(docs[0].lines["c"], 6)

    def test_command_targets(self):
        self.assertEqual(command_targets(["uvicorn", "app.main:app", "--port", "80"]), [("MODULE:app.main", "uvicorn")])
        self.assertEqual(command_targets("python -m worker.run --once"), [("MODULE:worker.run", "python -m")])
        self.assertEqual(command_targets("sh -c 'node dist/server.js'"), [("FILE:dist/server.js", "node")])
        self.assertEqual(command_targets("celery -A tasks.app worker"), [("MODULE:tasks.app", "celery")])
        self.assertEqual(command_targets("npm ci && python manage.py migrate"), [("FILE:manage.py", "python")])

    def test_hcl_blocks_ignore_braces_in_strings_and_heredocs(self):
        text = ('resource "aws_iam_policy" "p" {\n  policy = <<EOF\n{ "a": "}" }\nEOF\n  name = "x{y"\n}\n'
                'variable "v" {}\n')
        blocks = hcl_blocks(text)
        self.assertEqual([(b["type"], b["labels"]) for b in blocks],
                         [("resource", ["aws_iam_policy", "p"]), ("variable", ["v"])])

    def test_sql_statements_and_tables(self):
        stmts = sql_statements("-- c\nCREATE TABLE a (x int);\n/* b; */ INSERT INTO b SELECT * FROM a;\n"
                               "CREATE FUNCTION f() RETURNS void AS $$ BEGIN DELETE FROM c; END $$ LANGUAGE plpgsql;")
        self.assertEqual(len(stmts), 3)
        self.assertEqual(stmts[1][1], 3)
        self.assertEqual(tables_in("WITH t AS (SELECT 1) SELECT * FROM t JOIN public.users u ON 1=1"),
                         ({"users"}, set()))
        self.assertEqual(tables_in("UPDATE orders SET x = 1 WHERE id IN (SELECT id FROM items)"),
                         ({"items"}, {"orders"}))


class InfraScanTest(RepoTestCase):
    scan_on_setup = False

    def setUp(self):
        super().setUp()
        shutil.copytree(OVERLAY, self.root, dirs_exist_ok=True)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "infra")
        from promapkin.scanner import scan
        self.result = scan(self.ws)
        self.s = self.ws.store

    def edge(self, src, typ, dst):
        return any(e["dst"] == dst for e in self.s.edges(src=src, types=[typ]))

    def test_formats_are_recognised(self):
        parsers = self.result["parsers"]
        for fmt in ("dockerfile", "compose", "kubernetes", "terraform", "ci", "openapi", "sql", "prisma", "graphql",
                    "proto"):
            self.assertIn(fmt, parsers)
        self.assertEqual(self.result["errors"], [])

    def test_infrastructure_reaches_code(self):
        self.assertTrue(self.edge("container:Dockerfile", "runs", "module:searchapp.api.routes"))
        self.assertTrue(self.edge("k8s:search/workload/api", "runs", "module:searchapp.cli"))
        self.assertTrue(self.edge("ci:.github/workflows/ci.yml", "runs", "module:searchapp.cli"))
        self.assertTrue(self.edge("service:search", "routes_to", "k8s:search/workload/api"))
        self.assertTrue(self.edge("k8s:search/ingress/public", "routes_to", "service:search"))
        self.assertTrue(self.edge("service:api", "builds", "container:Dockerfile"))
        self.assertTrue(self.edge("service:api", "depends_on", "service:db"))
        self.assertTrue(self.edge("k8s:search/workload/api", "depends_on", "k8s:search/secret/search-secrets"))
        # Declared config and code reads meet on one node.
        self.assertTrue(self.edge("container:Dockerfile", "configures", "config:SEARCH_DB_PATH"))
        readers = [e["src"] for e in self.s.edges(dst="config:SEARCH_DB_PATH", types=["reads"])]
        self.assertIn("symbol:searchapp.config:Settings.__init__", readers)

    def test_terraform(self):
        lam = "tf:infra:aws_lambda_function.indexer"
        self.assertEqual(self.s.node(lam)["attrs"]["category"], "compute")
        self.assertEqual(self.s.node("tf:infra:aws_s3_bucket.docs")["attrs"]["category"], "storage")
        self.assertTrue(self.edge(lam, "depends_on", "tf:infra:aws_s3_bucket.docs"))
        self.assertTrue(self.edge(lam, "configures", "config:SEARCH_DB_PATH"))
        self.assertTrue(self.edge("tf:infra:module.net", "uses", "manifest:infra/modules/net/main.tf"))
        self.assertTrue(self.edge("tf:infra/modules/net:aws_vpc.main", "depends_on", "tf:infra/modules/net:var.cidr"))

    def test_data_model_links_code_ddl_and_orm(self):
        docs = self.s.node("table:docs")
        self.assertEqual(docs["owner_file"], "db/schema.sql")
        self.assertEqual([c["name"] for c in docs["attrs"]["columns"]], ["id", "text", "created_at"])
        self.assertTrue(self.edge("table:tags", "depends_on", "table:docs"))
        self.assertTrue(self.edge("table:tagged", "depends_on", "table:tags"))
        self.assertTrue(self.edge("symbol:searchapp.storage.index_repository:IndexRepository.add", "writes",
                                  "table:docs"))
        self.assertTrue(self.edge("symbol:searchapp.storage.index_repository:IndexRepository.find_all", "reads",
                                  "table:docs"))
        self.assertTrue(self.edge("symbol:searchapp.storage.orm:Document", "maps_to", "table:docs"))
        self.assertTrue(self.edge("table:post", "depends_on", "table:users"))     # Prisma relation + @@map
        flows = __import__("promapkin.graph", fromlist=["Graph"]).Graph(self.s).dataflow(
            "IndexRepository.add")["flows"]
        self.assertIn("table:docs", [f["sink"] for f in flows])

    def test_api_schemas_and_spec_coverage(self):
        self.assertTrue(self.edge("apispec:openapi.yaml#GET /search", "specifies", "api:GET /search"))
        self.assertTrue(self.edge("graphql:Query.search", "produces", "graphql:type:Doc"))
        self.assertTrue(self.edge("rpc:search.v1.Search/Query", "consumes", "proto:search.v1.QueryRequest"))
        self.assertEqual(self.s.node("proto:search.v1.QueryRequest")["attrs"]["fields"], ["q", "limit"])
        from promapkin.service import Service
        dm = Service(self.ws).data_model()
        self.assertEqual(dm["api_spec"]["unimplemented"], ["GET /missing"])
        self.assertEqual(dm["endpoints_by_protocol"], {"http": 2, "graphql": 3, "grpc": 2})

    def test_views_blueprint_and_mcp(self):
        from promapkin.service import Service
        svc = Service(self.ws)
        infra = svc.infrastructure()
        self.assertIn("BUCKET", infra["configuration"]["declared_but_never_read"])
        self.assertIn("storage", infra["cloud_resources"])
        self.assertIn("erDiagram", svc.diagram("er")["source"])
        self.assertIn("flowchart", svc.diagram("infrastructure")["source"])
        svc.generate_blueprint(["architecture"])
        md = (self.ws.blueprint_dir / "01-architecture.md").read_text()
        self.assertIn("## Infrastructure", md)
        self.assertIn("## Data model", md)
        self.assertTrue((self.ws.blueprint_dir / "diagrams" / "er.mmd").exists())
        from promapkin.mcp_server import TOOLS
        names = {t.name for t in TOOLS}
        self.assertTrue({"get_infrastructure", "get_data_model"} <= names)
        from promapkin.brief import brief
        self.assertIn("Infrastructure and schemas", brief(self.ws))

    def test_incremental_rescan_keeps_links(self):
        from promapkin.scanner import scan
        (self.root / "Dockerfile").write_text((self.root / "Dockerfile").read_text().replace(
            "searchapp.api.routes:app", "searchapp.cli:main"))
        scan(self.ws)
        self.assertTrue(self.edge("container:Dockerfile", "runs", "module:searchapp.cli"))
        self.assertFalse(self.edge("container:Dockerfile", "runs", "module:searchapp.api.routes"))
        self.assertTrue(self.edge("service:api", "builds", "container:Dockerfile"))


if __name__ == "__main__":
    unittest.main()
