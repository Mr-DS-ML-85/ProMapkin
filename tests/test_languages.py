"""Language coverage: every supported language extracts definitions, calls and imports, through tree-sitter
when it is installed and through the lexical fallback otherwise."""

import importlib.util
import unittest
from unittest import mock

from promapkin.extract import extractor_for, language_of

HAS_TS = importlib.util.find_spec("tree_sitter_language_pack") is not None

# language: (file name, source, expected definitions, expected call, expected import)
CASES = {
    'dart': ('src/sample.dart', "import 'package:foo/bar.dart';\nclass Repo extends Base implements I {\n  void save(int x) { helper(x); var r = Repo(); r.find(); }\n}\nint top() { return 1; }\n",
               ['Repo', 'save'], 'helper', 'package:foo/bar'),
    'lua': ('src/sample.lua', "local json = require('json')\nlocal M = {}\nfunction M.save(x)\n  helper(x)\n  obj:find(1)\nend\nlocal function top() return 1 end\n",
               ['save', 'top'], 'helper', 'json'),
    'elixir': ('src/sample.ex', 'defmodule App.Repo do\n  alias App.Db\n  import Ecto.Query\n  def save(x) do\n    helper(x)\n    Db.insert(x)\n  end\n  defp priv(y), do: y\nend\n',
               ['App.Repo', 'save', 'priv'], 'helper', 'App.Db'),
    'erlang': ('src/sample.erl', '-module(repo).\n-import(lists, [map/2]).\n-export([save/1]).\nsave(X) ->\n  helper(X),\n  db:insert(X).\n',
               ['save'], 'helper', 'lists'),
    'haskell': ('src/sample.hs', 'module Repo where\nimport Data.List (sort)\ndata User = User { name :: String }\nclass Store a where\n  put :: a -> IO ()\nsave :: Int -> IO ()\nsave x = helper x >> print x\n',
               ['User', 'Store', 'save'], 'helper', 'Data.List'),
    'ocaml': ('src/sample.ml', 'open Printf\nmodule Repo = struct\n  let save x = helper x; Db.insert x\nend\ntype user = { name : string }\nlet top () = 1\n',
               ['Repo', 'save'], 'helper', 'Printf'),
    'fsharp': ('src/sample.fs', 'module Repo\nopen System\ntype User(name: string) =\n  member this.Save(x) = helper x\nlet top x = Db.insert x\n',
               ['User', 'Save', 'top'], 'helper', 'System'),
    'clojure': ('src/sample.clj', '(ns app.repo (:require [app.db :as db]))\n(defn save [x] (helper x) (db/insert x))\n(defrecord User [name])\n',
               ['save', 'User'], 'helper', 'app.db'),
    'r': ('src/sample.r', 'library(dplyr)\nsave <- function(x) {\n  helper(x)\n  db$insert(x)\n}\n',
               ['save'], 'helper', 'dplyr'),
    'julia': ('src/sample.jl', 'module Repo\nusing LinearAlgebra\nimport Base: show\nstruct User\n  name::String\nend\nfunction save(x)\n  helper(x)\n  Db.insert(x)\nend\nend\n',
               ['Repo', 'User', 'save'], 'helper', 'LinearAlgebra'),
    'perl': ('src/sample.pm', 'package App::Repo;\nuse strict;\nuse App::Db;\nsub save {\n  my ($x) = @_;\n  helper($x);\n  App::Db->insert($x);\n}\n1;\n',
               ['save'], 'helper', 'App.Db'),
    'bash': ('src/sample.sh', 'source ./lib.sh\nsave() {\n  helper "$1"\n  curl http://x\n}\nfunction top { save 1; }\n',
               ['save', 'top'], 'helper', './lib'),
    'powershell': ('src/sample.ps1', "Import-Module Foo\nfunction Save-Item {\n  param($x)\n  Get-Helper $x\n}\nclass User { [void] Save() { Write-Host 'x' } }\n",
               ['Save-Item', 'User'], 'Get-Helper', 'Foo'),
    'zig': ('src/sample.zig', 'const std = @import("std");\nconst User = struct {\n  name: []const u8,\n  pub fn save(self: *User) void { helper(self); }\n};\npub fn main() void { std.debug.print("x", .{}); }\n',
               ['User', 'save', 'main'], 'helper', 'std'),
    'nim': ('src/sample.nim', 'import strutils\ntype User = object\n  name: string\nproc save(u: User) =\n  helper(u)\n  echo u.name\n',
               ['User', 'save'], 'helper', 'strutils'),
    'objc': ('src/sample.m', '#import "Foo.h"\n@interface Repo : NSObject\n- (void)save:(int)x;\n@end\n@implementation Repo\n- (void)save:(int)x { [self helper:x]; foo(x); }\n@end\n',
               ['Repo', 'save'], 'foo', 'Foo'),
    'elm': ('src/sample.elm', 'module Repo exposing (save)\nimport Html exposing (div)\ntype alias User = { name : String }\nsave : Int -> Int\nsave x = helper x\n',
               ['User', 'save'], 'helper', 'Html'),
    'solidity': ('src/sample.sol', 'pragma solidity ^0.8.0;\nimport "./Base.sol";\ncontract Token is Base {\n  function transfer(address to) public { helper(to); }\n}\ninterface IERC { function f() external; }\n',
               ['Token', 'transfer', 'IERC'], 'helper', None),
    'crystal': ('src/sample.cr', 'require "json"\nclass Repo < Base\n  def save(x)\n    helper(x)\n  end\nend\n',
               ['Repo', 'save'], 'helper', 'json'),
    'd': ('src/sample.d', 'import std.stdio;\nclass Repo : Base {\n  void save(int x) { helper(x); }\n}\nvoid main() { writeln(1); }\n',
               ['Repo', 'save', 'main'], 'helper', 'std.stdio'),
    'fortran': ('src/sample.f90', 'module repo\ncontains\n  subroutine save(x)\n    integer :: x\n    call helper(x)\n  end subroutine\n  function top() result(r)\n    r = 1\n  end function\nend module\n',
               ['save', 'top'], 'helper', None),
    'pascal': ('src/sample.pas', 'unit Repo;\ninterface\nuses SysUtils;\ntype TRepo = class(TObject)\n  procedure Save(x: Integer);\nend;\nimplementation\nprocedure TRepo.Save(x: Integer);\nbegin\n  Helper(x);\nend;\nend.\n',
               ['TRepo', 'Save'], 'Helper', 'SysUtils'),
    'matlab': ('src/sample.m', 'function r = save(x)\n  r = helper(x);\nend\n',
               ['save'], 'helper', None),
    'verilog': ('src/sample.v', 'module counter(input clk, output reg [3:0] q);\n  adder a1(.x(clk));\n  always @(posedge clk) q <= q + 1;\nendmodule\n',
               ['counter'], None, None),
    'vhdl': ('src/sample.vhd', 'library ieee;\nuse ieee.std_logic_1164.all;\nentity counter is\n  port (clk : in std_logic);\nend counter;\narchitecture rtl of counter is\nbegin\nend rtl;\n',
               ['counter'], None, None),
    'gleam': ('src/sample.gleam', 'import gleam/io\npub type User { User(name: String) }\npub fn save(x) {\n  helper(x)\n  io.println("x")\n}\n',
               ['User', 'save'], 'helper', None),
    'odin': ('src/sample.odin', 'package repo\nimport "core:fmt"\nUser :: struct { name: string }\nsave :: proc(u: User) {\n  helper(u)\n  fmt.println(u)\n}\n',
               ['User', 'save'], 'helper', None),
    'v': ('src/sample.v', 'module repo\nimport os\nstruct User { name string }\nfn (u User) save() { helper(u) }\nfn main() { os.exit(0) }\n',
               ['User', 'save', 'main'], 'helper', 'os'),
    'commonlisp': ('src/sample.lisp', '(defpackage :repo (:use :cl))\n(defun save (x) (helper x))\n(defclass user () ((name)))\n',
               ['save'], 'helper', None),
    'scheme': ('src/sample.scm', '(define (save x) (helper x))\n(define-record-type user (make-user name) user?)\n',
               ['save'], 'helper', None),
    'racket': ('src/sample.rkt', '#lang racket\n(require json)\n(define (save x) (helper x))\n(struct user (name))\n',
               ['save'], 'helper', 'json'),
    'cuda': ('src/sample.cu', '#include <cuda.h>\n__global__ void kernel(int *x) { helper(x); }\nint main() { kernel<<<1,1>>>(0); }\n',
               ['kernel', 'main'], 'helper', None),
    'glsl': ('src/sample.glsl', '#version 330\nuniform vec3 c;\nvec3 shade(vec3 n) { return normalize(n); }\nvoid main() { gl_FragColor = vec4(shade(c), 1.0); }\n',
               ['shade'], 'normalize', None),
    'hlsl': ('src/sample.hlsl', 'float4 shade(float3 n) { return float4(normalize(n), 1); }\nfloat4 main() : SV_Target { return shade(1); }\n',
               ['shade'], 'normalize', None),
    'apex': ('src/sample.cls', 'public class Repo extends Base {\n  public void save(Integer x) { helper(x); Account a = new Account(); }\n}\n',
               ['Repo', 'save'], 'helper', None),
}


def summary(facts):
    defs = {n["name"] for n in facts.nodes if n["kind"] not in ("module", "resource")} | {
        n["id"].rsplit(":", 1)[-1] for n in facts.nodes if n["kind"] not in ("module", "resource")}
    calls = {r["target"] for r in facts.refs if r["type"] == "calls"}
    imports = {r["target"] for r in facts.refs if r["type"] == "imports"}
    return defs, calls, imports


class LanguageDetectionTest(unittest.TestCase):
    def test_extensions_and_shared_suffixes(self):
        for lang, (path, source, *_rest) in CASES.items():
            self.assertEqual(language_of(path, source), lang, path)
        self.assertEqual(language_of("ui/App.vue"), "vue")
        self.assertEqual(language_of("build.gradle"), "groovy")
        self.assertGreaterEqual(len({language_of(p, s) for p, s, *_ in CASES.values()}), 35)


@unittest.skipUnless(HAS_TS, "tree-sitter-language-pack not installed (pip install 'promapkin[parsers]')")
class TreeSitterLanguagesTest(unittest.TestCase):
    def test_every_language(self):
        from promapkin.extract import treesitter_generic, treesitter_ext
        failures = []
        for lang, (path, source, defs, call, imp) in CASES.items():
            spec = treesitter_generic.SPECS.get(lang) or treesitter_ext.SPECS[lang]
            try:
                treesitter_ext._parser(spec.grammar)
            except Exception:  # noqa: BLE001 - grammar not downloadable here
                continue
            facts = extractor_for(lang)(path, source)
            got_defs, calls, imports = summary(facts)
            parser = facts.nodes[0]["attrs"].get("parser", "")
            if not parser.startswith("tree-sitter"):
                failures.append(f"{lang}: used {parser}")
            missing = [d for d in defs if d not in got_defs]
            if missing:
                failures.append(f"{lang}: missing definitions {missing} (got {sorted(got_defs)})")
            if call and not any(c == call or c.endswith("." + call) for c in calls):
                failures.append(f"{lang}: missing call {call} (got {sorted(calls)})")
            if imp and imp not in imports:
                failures.append(f"{lang}: missing import {imp} (got {sorted(imports)})")
        self.assertEqual(failures, [])


class LexicalFallbackTest(unittest.TestCase):
    def test_every_language_without_tree_sitter(self):
        failures = []
        with mock.patch("promapkin.toolchain.tree_sitter_available", return_value=False):
            for lang, (path, source, defs, call, imp) in CASES.items():
                facts = extractor_for(lang)(path, source)
                got_defs, calls, imports = summary(facts)
                if facts.nodes[0]["attrs"].get("parser") != "lexical":
                    failures.append(f"{lang}: not lexical")
                if not got_defs & set(defs):
                    failures.append(f"{lang}: no definitions found (expected some of {defs})")
        self.assertEqual(failures, [])

    def test_grammar_load_failure_falls_back_to_lexical(self):
        with mock.patch("promapkin.toolchain.tree_sitter_available", return_value=True), \
                mock.patch("promapkin.extract.treesitter_ext._parser", side_effect=RuntimeError("no network")):
            path, source, defs, *_ = CASES["elixir"]
            facts = extractor_for("elixir")(path, source)
        self.assertEqual(facts.nodes[0]["attrs"]["parser"], "lexical")
        self.assertTrue(any("lexical fallback" in e for e in facts.errors))
        self.assertIn("save", summary(facts)[0])


class EmbeddedComponentsTest(unittest.TestCase):
    def test_vue_script_block_keeps_line_numbers(self):
        src = ("<template>\n  <UserCard :user=\"u\" />\n</template>\n<script setup lang=\"ts\">\n"
               "import { load } from './api'\nfunction refresh() {\n  return load()\n}\n</script>\n")
        facts = extractor_for("vue")("ui/Profile.vue", src)
        fn = next(n for n in facts.nodes if n["name"] == "refresh")
        self.assertEqual(fn["line_start"], 6)
        self.assertIn("load", {r["target"] for r in facts.refs if r["type"] == "calls"})
        self.assertIn("UserCard", {r["target"] for r in facts.refs if r["type"] == "references"})


if __name__ == "__main__":
    unittest.main()
