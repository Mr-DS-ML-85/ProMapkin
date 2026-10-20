"""Graph insights: hubs ("god nodes"), communities, surprising cross-community connections,
and a self-contained interactive HTML explorer.

Surprising connections only use observed (parsed / resolved / runtime) evidence. Name matches,
ML similarity and documentation claims never count, so unrelated scripts that merely call the
same builtin are not reported as connected.
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from promapkin import evidence as ev
from promapkin.store import Store

CODE_EDGES = ["calls", "instantiates", "imports", "inherits", "implements", "depends_on", "references",
              "routes_to", "overrides", "invokes", "reads", "writes", "produces", "consumes"]
CODE_KINDS = ("module", "class", "function", "method", "endpoint", "test", "resource", "config", "event", "package")
TRUSTED = {ev.AST, ev.STATIC_RESOLVED, ev.LEXICAL, ev.RUNTIME, ev.MANIFEST}


def hubs(store: Store, limit: int = 25, kinds: tuple[str, ...] = ("class", "function", "method", "module")
         ) -> list[dict[str, Any]]:
    """Most-connected code nodes, counting distinct neighbours over code edges (docs excluded)."""
    marks = ",".join("?" * len(CODE_EDGES))
    fan_in: Counter[str] = Counter()
    fan_out: Counter[str] = Counter()
    for src, dst in store.conn.execute(f"SELECT DISTINCT src, dst FROM edges WHERE type IN ({marks})", CODE_EDGES):
        fan_out[src] += 1
        fan_in[dst] += 1
    rows = []
    kind_of = {r[0]: (r[1], r[2], r[3], r[4]) for r in store.conn.execute(
        f"SELECT id, kind, file, line_start, attrs FROM nodes WHERE kind IN ({','.join('?' * len(kinds))})", kinds)}
    for nid, (kind, file, line, attrs) in kind_of.items():
        if '"test": true' in (attrs or ""):
            continue
        degree = fan_in[nid] + fan_out[nid]
        if degree:
            rows.append({"id": nid, "kind": kind, "fan_in": fan_in[nid], "fan_out": fan_out[nid], "degree": degree,
                         "location": f"{file}:{line}" if file and line else file})
    rows.sort(key=lambda r: (-r["fan_in"], -r["degree"], r["id"]))
    return rows[:limit]


def communities(store: Store, rt: Any = None) -> list[dict[str, Any]]:
    from promapkin.ml import graph_ml
    model = None
    if rt is not None and rt.classifier.mode != "heuristic":
        model = graph_ml.graph_model_name(rt.embedder.name)
    return graph_ml.cluster_modules(store, model)


def _module_of(store: Store, nid: str, cache: dict[str, str | None]) -> str | None:
    if nid in cache:
        return cache[nid]
    out = None
    if nid.startswith("module:"):
        out = nid
    elif nid.startswith("symbol:") and nid.count(":") >= 2:
        out = "module:" + nid.split(":")[1]
    cache[nid] = out
    return out


def surprises(store: Store, rt: Any = None, limit: int = 20) -> dict[str, Any]:
    """Observed edges that cross community boundaries rarely (few edges between the two communities)."""
    comms = communities(store, rt)
    member = {m: i for i, c in enumerate(comms) for m in c["modules"]}
    cache: dict[str, str | None] = {}
    tests = {n["id"] for n in store.nodes("module") if n["attrs"].get("test")}
    between: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    marks = ",".join("?" * len(CODE_EDGES))
    for e in store.edges(where=f"type IN ({marks})", params=CODE_EDGES):
        if e["evidence"] not in TRUSTED or e["status"] != "verified":
            continue
        a, b = _module_of(store, e["src"], cache), _module_of(store, e["dst"], cache)
        if not a or not b or a == b or a in tests or b in tests:
            continue
        ca, cb = member.get(a), member.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        between[(min(ca, cb), max(ca, cb))].append(e)
    fan_in: Counter[str] = Counter(dst for (dst,) in store.conn.execute(
        f"SELECT dst FROM edges WHERE type IN ({marks})", CODE_EDGES))
    rows = []
    for (ca, cb), es in between.items():
        # The least expected link: few edges between the communities, and a target that is not a hub
        # everybody imports (a link to `fastapi` from a tutorial is not news).
        e = min(es, key=lambda x: fan_in[x["dst"]])
        rows.append({"source": e["src"], "type": e["type"], "target": e["dst"], "evidence": e["evidence"],
                     "at": f"{e['file']}:{e['line']}" if e.get("file") else None,
                     "communities": [comms[ca]["component"], comms[cb]["component"]],
                     "edges_between_communities": len(es), "target_fan_in": fan_in[e["dst"]],
                     "why": f"only {len(es)} observed edge(s) connect these two communities"})
    rows.sort(key=lambda r: (r["edges_between_communities"] * (1 + r["target_fan_in"]), r["source"]))
    return {"communities": len(comms), "connections": rows[:limit],
            "policy": "observed evidence only (AST, resolved imports, lexical, runtime); name matches, ML "
                      "similarity and documentation claims are excluded"}


# ------------------------------------------------------------------------------------- HTML explorer

def export_html(store: Store, out: Path, rt: Any = None, max_nodes: int = 1500, title: str = "Promapkin graph"
                ) -> dict[str, Any]:
    comms = communities(store, rt)
    member = {m: i for i, c in enumerate(comms) for m in c["modules"]}
    marks = ",".join("?" * len(CODE_EDGES))
    degree: Counter[str] = Counter()
    edges = []
    for e in store.edges(where=f"type IN ({marks})", params=CODE_EDGES):
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
        edges.append(e)
    kinds = ",".join("?" * len(CODE_KINDS))
    nodes = {r["id"]: r for r in store.nodes(where=f"kind IN ({kinds})", params=CODE_KINDS)}
    keep = sorted((n for n in nodes if degree[n]), key=lambda n: -degree[n])[:max_nodes]
    keep_set = set(keep)
    cache: dict[str, str | None] = {}
    data_nodes = []
    for nid in keep:
        n = nodes[nid]
        mod = _module_of(store, nid, cache)
        data_nodes.append({"id": nid, "label": n["name"], "kind": n["kind"], "degree": degree[nid],
                           "group": member.get(mod, -1) if mod else -1,
                           "loc": f"{n['file']}:{n['line_start']}" if n["file"] and n["line_start"] else (n["file"] or ""),
                           "doc": (n["text"] or "")[:200]})
    data_edges = []
    seen = set()
    for e in edges:
        if e["src"] in keep_set and e["dst"] in keep_set:
            key = (e["src"], e["dst"], e["type"])
            if key in seen:
                continue
            seen.add(key)
            data_edges.append({"s": e["src"], "t": e["dst"], "type": e["type"], "ev": e["evidence"],
                               "cls": ev.classify(e["evidence"])})
    payload = {"nodes": data_nodes, "edges": data_edges,
               "groups": [c["component"] for c in comms], "title": title}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(payload).replace("</", "<\\/")))
    return {"path": str(out), "nodes": len(data_nodes), "edges": len(data_edges), "communities": len(comms),
            "truncated": len(nodes) > len(data_nodes)}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--fg:#1c1b19;--muted:#6b675f;--panel:#fffdf8;--line:#d9d4c7;--accent:#8a6d1d}
@media (prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ece8df;--muted:#a39d90;--panel:#1d1c19;--line:#34322c;--accent:#d8b45a}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0;letter-spacing:.02em}header input,header select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
header label{color:var(--muted);font-size:12px;display:flex;gap:6px;align-items:center}
main{flex:1;display:flex;min-height:0}canvas{flex:1;display:block;cursor:grab;min-width:0}
aside{width:320px;max-width:40vw;border-left:1px solid var(--line);background:var(--panel);padding:14px;overflow:auto}
aside h2{font-size:14px;margin:0 0 4px}aside .k{color:var(--muted);font-size:12px}aside ul{padding-left:16px;margin:6px 0}
aside li{margin:2px 0;cursor:pointer}aside li:hover{color:var(--accent)}.tag{font-size:11px;color:var(--muted)}
@media (max-width:700px){main{flex-direction:column}aside{width:auto;max-width:none;height:40vh;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1>__TITLE__</h1><input id="q" placeholder="Search symbol…" size="22">
<label>Kind <select id="kind"><option value="">all</option></select></label>
<label><input type="checkbox" id="observed"> observed edges only</label><span class="k" id="stats"></span></header>
<main><canvas id="c"></canvas><aside id="info"><h2>Explore the graph</h2><p class="k">Drag to pan, scroll to zoom, click a node for its relationships and evidence. Dashed edges are inferred.</p></aside></main>
<script>
const D=__DATA__;
const cv=document.getElementById('c'),cx=cv.getContext('2d'),info=document.getElementById('info');
const palette=['#8a6d1d','#3d6fa5','#a3432f','#4b8a5a','#7b4fa0','#b07a2a','#2f8a8a','#9a3f6c','#5f6b2f','#47528f'];
const byId=new Map();D.nodes.forEach((n,i)=>{const a=i*2.399963,r=12*Math.sqrt(i);n.x=Math.cos(a)*r;n.y=Math.sin(a)*r;n.vx=0;n.vy=0;byId.set(n.id,n)});
const E=D.edges.filter(e=>byId.has(e.s)&&byId.has(e.t)).map(e=>({...e,a:byId.get(e.s),b:byId.get(e.t)}));
const kinds=[...new Set(D.nodes.map(n=>n.kind))].sort();const ks=document.getElementById('kind');kinds.forEach(k=>{const o=document.createElement('option');o.value=o.textContent=k;ks.appendChild(o)});
document.getElementById('stats').textContent=`${D.nodes.length} nodes · ${E.length} edges · ${D.groups.length} communities`;
let view={x:0,y:0,k:1},sel=null,hover=null,filterKind='',observedOnly=false,query='';
function resize(){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio}addEventListener('resize',resize);resize();
function visible(n){return(!filterKind||n.kind===filterKind)}
function step(){const N=D.nodes;const cool=Math.max(.15,1-ticks/400);
 for(let i=0;i<N.length;i++){const a=N[i];for(let j=i+1;j<Math.min(N.length,i+80);j++){const b=N[j];let dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+1;if(d2>90000)continue;const f=80/d2;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f}}
 for(const e of E){if(e.a===e.b)continue;const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,d=Math.sqrt(dx*dx+dy*dy)+.01,f=(d-50)*.02/d;
  const fa=f/Math.sqrt(e.a.degree),fb=f/Math.sqrt(e.b.degree);e.a.vx+=dx*fa;e.a.vy+=dy*fa;e.b.vx-=dx*fb;e.b.vy-=dy*fb}
 for(const n of N){n.vx=(n.vx-n.x*.002)*.8*cool;n.vy=(n.vy-n.y*.002)*.8*cool;const sp=Math.hypot(n.vx,n.vy);if(sp>12){n.vx*=12/sp;n.vy*=12/sp}
  if(!Number.isFinite(n.vx)||!Number.isFinite(n.vy)){n.vx=0;n.vy=0}n.x+=n.vx;n.y+=n.vy}}
function toScreen(n){return[(n.x*view.k+view.x)*devicePixelRatio+cv.width/2,(n.y*view.k+view.y)*devicePixelRatio+cv.height/2]}
function draw(){const css=getComputedStyle(document.documentElement);cx.clearRect(0,0,cv.width,cv.height);cx.lineWidth=devicePixelRatio*.6;
 const nb=sel?new Set(E.filter(e=>e.a===sel||e.b===sel).flatMap(e=>[e.a,e.b])):null;
 for(const e of E){if(!visible(e.a)||!visible(e.b))continue;if(observedOnly&&e.cls!=='observed')continue;const[x1,y1]=toScreen(e.a),[x2,y2]=toScreen(e.b);
  cx.globalAlpha=nb?(nb.has(e.a)&&nb.has(e.b)&&(e.a===sel||e.b===sel)?.9:.05):.18;cx.strokeStyle=css.getPropertyValue('--muted');cx.setLineDash(e.cls==='observed'?[]:[4*devicePixelRatio,3*devicePixelRatio]);cx.beginPath();cx.moveTo(x1,y1);cx.lineTo(x2,y2);cx.stroke()}
 cx.setLineDash([]);for(const n of D.nodes){if(!visible(n))continue;const[x,y]=toScreen(n),r=(2+Math.sqrt(n.degree))*devicePixelRatio*Math.min(1.6,Math.max(.6,view.k));
  const match=query&&n.label.toLowerCase().includes(query);cx.globalAlpha=nb?(nb.has(n)?1:.15):(query?(match?1:.2):.9);cx.fillStyle=n.group>=0?palette[n.group%palette.length]:'#888';cx.beginPath();cx.arc(x,y,r,0,7);cx.fill();
  if(n===sel||n===hover||match||(view.k>1.6&&n.degree>4)){cx.globalAlpha=1;cx.fillStyle=css.getPropertyValue('--fg');cx.font=`${11*devicePixelRatio}px system-ui`;cx.fillText(n.label,x+r+3,y+3)}}cx.globalAlpha=1}
let ticks=0;function fit(){let m=1;for(const n of D.nodes)m=Math.max(m,Math.abs(n.x),Math.abs(n.y));view.k=Math.min(2,0.9*Math.min(cv.clientWidth,cv.clientHeight)/2/m)}
function loop(){if(ticks<400){step();ticks++;if(ticks%40===0&&!sel)fit()}draw();requestAnimationFrame(loop)}loop();
function pick(ev){const r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left)*devicePixelRatio,my=(ev.clientY-r.top)*devicePixelRatio;let best=null,bd=1e9;
 for(const n of D.nodes){if(!visible(n))continue;const[x,y]=toScreen(n),d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=n}}return bd<(14*devicePixelRatio)**2?best:null}
let drag=null;cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y,moved:false}});
addEventListener('mouseup',e=>{if(drag&&!drag.moved){select(pick(e))}drag=null});
cv.addEventListener('mousemove',e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;view.x=drag.vx+dx;view.y=drag.vy+dy}else hover=pick(e)});
cv.addEventListener('wheel',e=>{e.preventDefault();view.k*=e.deltaY<0?1.12:1/1.12},{passive:false});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function select(n){sel=n;if(!n){return}const out=E.filter(e=>e.a===n),inn=E.filter(e=>e.b===n);
 const li=(e,o)=>`<li data-id="${esc(o.id)}">${esc(e.type)} <b>${esc(o.label)}</b> <span class="tag">[${esc(e.ev)}]</span></li>`;
 info.innerHTML=`<h2>${esc(n.label)}</h2><div class="k">${esc(n.kind)} · ${esc(n.loc)}</div><div class="k">community: ${esc(D.groups[n.group]||'—')} · degree ${n.degree}</div>${n.doc?`<p>${esc(n.doc)}</p>`:''}
 <h3>Outgoing (${out.length})</h3><ul>${out.slice(0,80).map(e=>li(e,e.b)).join('')}</ul><h3>Incoming (${inn.length})</h3><ul>${inn.slice(0,80).map(e=>li(e,e.a)).join('')}</ul>`;
 info.querySelectorAll('li').forEach(l=>l.onclick=()=>{const t=byId.get(l.dataset.id);select(t);view.x=-t.x*view.k;view.y=-t.y*view.k})}
document.getElementById('q').oninput=e=>{query=e.target.value.toLowerCase();const m=D.nodes.find(n=>n.label.toLowerCase()===query);if(m){select(m);view.x=-m.x*view.k;view.y=-m.y*view.k}};
ks.onchange=e=>filterKind=e.target.value;document.getElementById('observed').onchange=e=>observedOnly=e.target.checked;
</script></body></html>
"""
