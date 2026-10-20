#!/usr/bin/env node
/*
 * Promapkin TypeScript/JavaScript extractor.
 *
 * Uses the TypeScript compiler API (parser + type checker) to extract declarations, imports,
 * call/new targets resolved through types, inheritance, routes, environment reads and event
 * wiring. Reads {root, files, typescript} JSON on stdin and writes one JSON document to stdout.
 * It never executes repository code: it only parses and type-checks it.
 */
"use strict";

const fs = require("fs");
const path = require("path");

function readStdin() {
  return fs.readFileSync(0, "utf8");
}

const input = JSON.parse(readStdin());
const ts = require(input.typescript);
const root = path.resolve(input.root);
const wanted = new Set(input.files.map((f) => path.resolve(root, f)));

function rel(p) {
  return path.relative(root, p).split(path.sep).join("/");
}

function moduleName(relPath) {
  let p = relPath.replace(/\.(d\.ts|tsx?|jsx?|mjs|cjs)$/, "");
  if (p.endsWith("/index")) p = p.slice(0, -"/index".length);
  else if (p === "index") p = ".";
  return p;
}

function inRepo(fileName) {
  const abs = path.resolve(fileName);
  return abs.startsWith(root + path.sep) && !abs.includes(`${path.sep}node_modules${path.sep}`);
}

function packageOf(fileName) {
  const norm = fileName.split(path.sep).join("/");
  // The TypeScript standard library (lib.*.d.ts) describes the runtime, not a dependency.
  if (/node_modules\/typescript\/lib\//.test(norm) || /\/lib\.[^/]*\.d\.ts$/.test(norm)) return null;
  const m = norm.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/);
  if (!m) return null;
  return m[1].startsWith("@types/") ? m[1].slice("@types/".length) : m[1];
}

// Compiler options: the repository's tsconfig if present, forced to analysis-only settings.
let options = {
  allowJs: true, checkJs: false, noEmit: true, skipLibCheck: true, jsx: ts.JsxEmit.Preserve,
  target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler || ts.ModuleResolutionKind.NodeJs,
  esModuleInterop: true, resolveJsonModule: false, types: [],
};
const cfgPath = ts.findConfigFile(root, ts.sys.fileExists, "tsconfig.json");
if (cfgPath && path.dirname(cfgPath) === root) {
  const cfg = ts.readConfigFile(cfgPath, ts.sys.readFile);
  if (!cfg.error) {
    const parsed = ts.parseJsonConfigFileContent(cfg.config, ts.sys, root);
    options = Object.assign({}, parsed.options, { allowJs: true, noEmit: true, skipLibCheck: true, types: [] });
  }
}

const program = ts.createProgram([...wanted], options);
const checker = program.getTypeChecker();

function lineOf(sf, pos) {
  return sf.getLineAndCharacterOfPosition(pos).line + 1;
}

function docOf(node) {
  const docs = ts.getJSDocCommentsAndTags ? ts.getJSDocCommentsAndTags(node) : [];
  const parts = [];
  for (const d of docs) {
    if (typeof d.comment === "string") parts.push(d.comment);
    else if (Array.isArray(d.comment)) parts.push(d.comment.map((c) => c.text || "").join(""));
  }
  return parts.join(" ").trim() || null;
}

// Qualified name of a declaration within its file: Class.method, function, Class.
function qualOf(decl) {
  const names = [];
  let n = decl;
  while (n && !ts.isSourceFile(n)) {
    if ((ts.isClassDeclaration(n) || ts.isClassExpression(n) || ts.isInterfaceDeclaration(n) ||
         ts.isFunctionDeclaration(n) || ts.isMethodDeclaration(n) || ts.isMethodSignature(n) ||
         ts.isPropertyDeclaration(n) || ts.isGetAccessor(n) || ts.isSetAccessor(n) ||
         ts.isEnumDeclaration(n) || ts.isModuleDeclaration(n)) && n.name) {
      names.unshift(n.name.getText());
    } else if (ts.isConstructorDeclaration(n)) {
      names.unshift("constructor");
    } else if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name)) {
      names.unshift(n.name.text);
    }
    n = n.parent;
  }
  return names.join(".");
}

function targetOfSymbol(sym) {
  if (!sym) return null;
  if (sym.flags & ts.SymbolFlags.Alias) {
    try { sym = checker.getAliasedSymbol(sym); } catch (e) { /* unresolved alias */ }
  }
  const decl = (sym.declarations || [])[0] || sym.valueDeclaration;
  if (!decl) return null;
  const sf = decl.getSourceFile();
  if (inRepo(sf.fileName)) {
    return { file: rel(sf.fileName), module: moduleName(rel(sf.fileName)), qual: qualOf(decl) || sym.getName() };
  }
  const pkg = packageOf(sf.fileName);
  return pkg ? { package: pkg, name: sym.getName() } : null;
}

function resolveCall(node) {
  // Prefer the resolved signature's declaration (handles overloads, generics, this.x.y()).
  try {
    const sig = checker.getResolvedSignature(node);
    if (sig && sig.declaration && !ts.isJSDocSignature(sig.declaration)) {
      const decl = sig.declaration;
      const sf = decl.getSourceFile();
      if (inRepo(sf.fileName)) {
        const q = qualOf(decl);
        if (q) return { file: rel(sf.fileName), module: moduleName(rel(sf.fileName)), qual: q, via: "signature" };
      } else {
        const pkg = packageOf(sf.fileName);
        if (pkg) return { package: pkg, name: decl.name ? decl.name.getText() : "", via: "signature" };
      }
    }
  } catch (e) { /* fall through */ }
  const expr = node.expression;
  const target = ts.isPropertyAccessExpression(expr) ? expr.name : expr;
  const t = targetOfSymbol(checker.getSymbolAtLocation(target));
  if (t) t.via = "symbol";
  return t;
}

const ROUTE_VERBS = new Set(["get", "post", "put", "delete", "patch", "all", "head", "options"]);
const EVENT_ON = new Set(["on", "once", "addListener", "addEventListener", "subscribe"]);
const EVENT_EMIT = new Set(["emit", "dispatchEvent", "publish", "trigger"]);
const out = { files: {}, typescript: ts.version };

for (const sf of program.getSourceFiles()) {
  if (!wanted.has(path.resolve(sf.fileName))) continue;
  const file = rel(sf.fileName);
  const mod = moduleName(file);
  const res = { module: mod, lines: sf.getLineAndCharacterOfPosition(sf.end).line + 1, doc: null, decls: [],
                imports: [], calls: [], inherits: [], routes: [], env: [], events: [], exports: [] };
  const first = sf.statements[0];
  if (first) res.doc = docOf(first);
  const stack = [];

  function owner() {
    return stack.length ? stack[stack.length - 1] : null;
  }

  function addDecl(node, kind, name, extra) {
    const qual = qualOf(node) || name;
    res.decls.push(Object.assign({
      qual, kind, name, line_start: lineOf(sf, node.getStart(sf)), line_end: lineOf(sf, node.end),
      doc: docOf(node),
      exported: !!(ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Export),
    }, extra || {}));
    return qual;
  }

  function params(fn) {
    return (fn.parameters || []).map((p) => ({
      name: p.name.getText(sf),
      type: p.type ? p.type.getText(sf) : (() => {
        try { return checker.typeToString(checker.getTypeAtLocation(p)); } catch (e) { return null; }
      })(),
    }));
  }

  function returns(fn) {
    if (fn.type) return fn.type.getText(sf);
    try {
      const sig = checker.getSignatureFromDeclaration(fn);
      return sig ? checker.typeToString(checker.getReturnTypeOfSignature(sig)) : null;
    } catch (e) { return null; }
  }

  function visit(node) {
    let pushed = false;
    if (ts.isImportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      const spec = node.moduleSpecifier.text;
      const resolved = ts.resolveModuleName(spec, sf.fileName, options, ts.sys).resolvedModule;
      const imp = { spec, line: lineOf(sf, node.getStart(sf)) };
      if (resolved && inRepo(resolved.resolvedFileName)) imp.module = moduleName(rel(resolved.resolvedFileName));
      else imp.package = spec.startsWith("node:") ? spec.slice(5) : spec;
      res.imports.push(imp);
    } else if (ts.isClassDeclaration(node) || ts.isClassExpression(node) || ts.isInterfaceDeclaration(node)) {
      const name = node.name ? node.name.text : "default";
      const methods = [];
      for (const m of node.members) {
        if ((ts.isMethodDeclaration(m) || ts.isMethodSignature(m)) && m.name) methods.push(m.name.getText(sf));
        if (ts.isConstructorDeclaration(m)) methods.push("constructor");
      }
      const isAbstract = ts.isInterfaceDeclaration(node) ||
        !!(ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Abstract);
      const qual = addDecl(node, "class", name, { methods, abstract: isAbstract,
                                                 interface: ts.isInterfaceDeclaration(node) });
      for (const clause of node.heritageClauses || []) {
        const rel_ = clause.token === ts.SyntaxKind.ExtendsKeyword ? "inherits" : "implements";
        for (const t of clause.types) {
          const tgt = targetOfSymbol(checker.getSymbolAtLocation(t.expression));
          res.inherits.push({ from: qual, type: rel_, text: t.expression.getText(sf), target: tgt,
                              line: lineOf(sf, t.getStart(sf)) });
        }
      }
      stack.push(qual); pushed = true;
    } else if (ts.isFunctionDeclaration(node) && node.name) {
      stack.push(addDecl(node, "function", node.name.text,
                         { params: params(node), returns: returns(node), async: !!(node.modifiers || []).some(
                           (m) => m.kind === ts.SyntaxKind.AsyncKeyword) }));
      pushed = true;
    } else if (ts.isMethodSignature(node) && node.name) {
      addDecl(node, "method", node.name.getText(sf), { params: params(node), returns: returns(node),
                                                      abstract: true, public: true });
    } else if ((ts.isMethodDeclaration(node) || ts.isConstructorDeclaration(node) || ts.isGetAccessor(node) ||
                ts.isSetAccessor(node)) && (node.body || ts.isMethodDeclaration(node))) {
      const name = ts.isConstructorDeclaration(node) ? "constructor" : node.name.getText(sf);
      const injected = ts.isConstructorDeclaration(node) ? node.parameters.filter((p) => p.type).map((p) => {
        const tgt = targetOfSymbol(checker.getSymbolAtLocation(
          ts.isTypeReferenceNode(p.type) ? p.type.typeName : p.type));
        return { name: p.name.getText(sf), type: p.type.getText(sf), target: tgt };
      }) : [];
      const flags = ts.getCombinedModifierFlags(node);
      stack.push(addDecl(node, "method", name, {
        params: params(node), returns: returns(node), injected,
        public: !(flags & (ts.ModifierFlags.Private | ts.ModifierFlags.Protected)) && !name.startsWith("#"),
      }));
      pushed = true;
    } else if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer &&
               (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))) {
      stack.push(addDecl(node, "function", node.name.text,
                         { params: params(node.initializer), returns: returns(node.initializer) }));
      pushed = true;
    } else if (ts.isCallExpression(node) || ts.isNewExpression(node)) {
      const line = lineOf(sf, node.getStart(sf));
      const tgt = resolveCall(node);
      res.calls.push({ from: owner(), kind: ts.isNewExpression(node) ? "new" : "call",
                       text: node.expression.getText(sf).slice(0, 160), target: tgt, line });
      const expr = node.expression;
      if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(expr)) {
        const verb = expr.name.text;
        const args = node.arguments;
        if (ROUTE_VERBS.has(verb) && args.length >= 2 && ts.isStringLiteralLike(args[0]) &&
            args[0].text.startsWith("/")) {
          const handlers = [];
          for (const a of args.slice(1)) {
            const t = ts.isIdentifier(a) || ts.isPropertyAccessExpression(a)
              ? targetOfSymbol(checker.getSymbolAtLocation(ts.isPropertyAccessExpression(a) ? a.name : a)) : null;
            handlers.push({ text: a.getText(sf).slice(0, 80), target: t,
                            inline: ts.isArrowFunction(a) || ts.isFunctionExpression(a) });
          }
          res.routes.push({ method: verb === "all" ? "ANY" : verb.toUpperCase(), route: args[0].text,
                            router: expr.expression.getText(sf), handlers, line, owner: owner() });
        }
        if ((EVENT_ON.has(verb) || EVENT_EMIT.has(verb)) && args.length >= 1 && ts.isStringLiteralLike(args[0])) {
          let handler = null;
          if (EVENT_ON.has(verb) && args[1] && (ts.isIdentifier(args[1]) || ts.isPropertyAccessExpression(args[1]))) {
            handler = targetOfSymbol(checker.getSymbolAtLocation(
              ts.isPropertyAccessExpression(args[1]) ? args[1].name : args[1]));
          }
          res.events.push({ event: args[0].text, role: EVENT_ON.has(verb) ? "subscribe" : "emit",
                            owner: owner(), handler, line });
        }
      }
    } else if (ts.isPropertyAccessExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
               node.expression.getText(sf) === "process.env") {
      res.env.push({ key: node.name.text, owner: owner(), line: lineOf(sf, node.getStart(sf)) });
    } else if (ts.isElementAccessExpression(node) && node.expression.getText(sf) === "process.env" &&
               ts.isStringLiteralLike(node.argumentExpression)) {
      res.env.push({ key: node.argumentExpression.text, owner: owner(), line: lineOf(sf, node.getStart(sf)) });
    }
    ts.forEachChild(node, visit);
    if (pushed) stack.pop();
  }

  ts.forEachChild(sf, visit);
  out.files[file] = res;
}

process.stdout.write(JSON.stringify(out));
