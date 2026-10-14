"""
Text → Tree → Graph → SVG  Multi-Agent Pipeline
================================================
Agents  : TreeBuilder · SVG_Engineer · Critic
          (Coordinator kept as thin watchdog only)

Ablation flags (CLI):
  --generation_mode   one_shot | autoregressive
  --reward_ablation   full | no_clip | no_hps | no_clip_no_hps | llm_only
  --no_textgrad       disable TGD, fall back to plain LLM feedback retry
"""
# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import os, json, io, re, time, sys, shutil, base64, argparse, hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests as _requests
from openai import OpenAI, APIError
from PIL import Image
from svgpathtools import parse_path
from playwright.sync_api import sync_playwright
import atexit

try:
    import textgrad as tg
    from textgrad.engine.local_model_openai_api import ChatExternalClient
    _TEXTGRAD_AVAILABLE = True
except Exception as exc:
    _TEXTGRAD_AVAILABLE = False
    print(f"[Warning] textgrad unavailable ({exc}) – TGD will be disabled.")

try:
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# 1.  Global configuration
# ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE     = "https://openrouter.ai/api/v1"
MODEL               = os.environ.get("STRUCTSVG_MODEL", "gemini-3-flash-preview")

# TextGrad backward engine: GPT-4o routed through OpenRouter.
# OpenRouter model string for GPT-4o: "openai/gpt-4o"
# Both forward and backward use the same OPENROUTER_API_KEY — no separate key needed.
TEXTGRAD_BACKWARD_MODEL = os.environ.get("STRUCTSVG_BACKWARD_MODEL", MODEL)

SCORER_URL          = os.environ.get("STRUCTSVG_SCORER_URL", "http://localhost:5001/score")

WORKSPACE_DIR   = os.path.abspath(".workspace")
PUBLISHED_DIR   = os.path.abspath("generated_svgs")
MEMORY_FILE     = os.path.join(WORKSPACE_DIR, "global_memory.json")

# Workspace file names
SEMANTIC_TREE_FILE      = "semantic_tree.json"
PRIMITIVE_GRAPH_FILE    = "primitive_graph.json"
CANDIDATE_MANIFEST_FILE = "candidate_manifest.json"
BEST_SVG_TREE_FILE      = "best_svg_tree.json"
BEST_SCORE_FILE         = "best_score.json"
FAIL_COUNTER_FILE       = "fail_counter.json"
FINISH_SUMMARY_FILE     = "finish_summary.json"
HISTORY_QUEUE_FILE      = os.path.join(WORKSPACE_DIR, "history_queue.json")

MAX_PATCH_RETRIES = 3   # shared across Tier-1 and Tier-2, each retry costs 1

RUNTIME_CLEAN_DIRS = [
    ".workspace",
    "generated_svgs",
    ".debug_logs",
    "critic_debug",
    "logs",
    "__pycache__",
]

def clean_runtime_dirs() -> None:
    """Remove generated runtime/cache folders while preserving results and models."""
    root = Path.cwd().resolve()
    keep = {root / "results", root / "models"}
    for rel in RUNTIME_CLEAN_DIRS:
        path = (root / rel).resolve()
        if path in keep or root not in path.parents:
            print(f"[Clean] Skip unsafe path: {path}")
            continue
        if path.exists():
            shutil.rmtree(path)
            print(f"[Clean] Removed {path.relative_to(root)}")

# Files each agent can read (shared + per-agent)
WORKSPACE_CONTEXT_SHARED_FILES = [
    "user_task.txt",
    "global_memory.json",
    SEMANTIC_TREE_FILE,
    PRIMITIVE_GRAPH_FILE,
    "current_best_icon.svg",
    BEST_SVG_TREE_FILE,
    FINISH_SUMMARY_FILE,
]

WORKSPACE_CONTEXT_AGENT_FILES: Dict[str, List[str]] = {
    "Coordinator":   [BEST_SCORE_FILE, FAIL_COUNTER_FILE],
    "TreeBuilder":   [],
    "SVG_Engineer":  ["history_queue.json", CANDIDATE_MANIFEST_FILE],
    "Critic":        ["history_queue.json", CANDIDATE_MANIFEST_FILE],
}

# Write-permission whitelist per agent
AGENT_WRITE_RULES: Dict[str, set] = {
    "Coordinator":  set(),
    "TreeBuilder":  {SEMANTIC_TREE_FILE},
    "SVG_Engineer": set(),
    "Critic":       set(),
}

# ─────────────────────────────────────────────────────────────
# 2.  API clients
# ─────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_API_KEY or "not-configured",
)

TEXTGRAD_FORWARD_ENGINE = None
PROMPT_OVERRIDES_FILE = "tgd_prompt_overrides.json"
PROMPT_OVERRIDES_PREV_FILE = "tgd_prompt_overrides_prev.json"
PROMPT_OVERRIDE_METRIC_FILE = "tgd_prompt_metric.json"

def _setup_textgrad_engines() -> bool:
    """
    Initialise TextGrad forward + backward engines, both via OpenRouter.

    Forward  engine : Gemini Flash  (cheap, used for loss *computation* calls)
    Backward engine : GPT-4o        (strong model needed for gradient quality)

    Both share the single OPENROUTER_API_KEY — no separate OpenAI key required.
    OpenRouter model string for GPT-4o is "openai/gpt-4o".
    """
    if not _TEXTGRAD_AVAILABLE:
        return False
    if not OPENROUTER_API_KEY:
        print("[Warning] OPENROUTER_API_KEY not set – TextGrad disabled.")
        return False
    try:
        # Both engines point at the same OpenRouter base URL / key,
        # but use different model strings so billing/routing is correct.
        openrouter_client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)

        forward_engine  = ChatExternalClient(client=openrouter_client, model_string=MODEL)
        backward_engine = ChatExternalClient(client=openrouter_client,
                                             model_string=TEXTGRAD_BACKWARD_MODEL)

        # TextGrad API: set_backward_engine called twice —
        # first call sets the *forward* engine used inside TextLoss,
        # second call with override=True pins the *backward* (gradient) engine.
        tg.set_backward_engine(backward_engine, override=True)
        global TEXTGRAD_FORWARD_ENGINE
        TEXTGRAD_FORWARD_ENGINE = forward_engine
        print(f"[TextGrad] Forward={MODEL}  Backward={TEXTGRAD_BACKWARD_MODEL}  (both via OpenRouter)")
        return True
    except Exception as e:
        print(f"[Warning] TextGrad engine setup failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# 3.  Playwright singleton  (SVG → PNG)
# ─────────────────────────────────────────────────────────────
class PlaywrightSingleton:
    _instance = None

    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser    = self.playwright.chromium.launch()
        self.context    = self.browser.new_context(device_scale_factor=2)
        self.page       = self.context.new_page()
        atexit.register(self.close)

    @classmethod
    def get_page(cls):
        if cls._instance is None:
            print("🚀 [System] Starting Playwright singleton...")
            cls._instance = cls()
        return cls._instance.page

    def close(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass


def svg_to_png_with_browser(svg_path: str, png_path: str,
                             width: int = 256, height: int = 256):
    page = PlaywrightSingleton.get_page()
    page.goto(Path(svg_path).absolute().as_uri())
    el = page.locator("svg")
    el.evaluate(f"(e)=>{{e.style.width='{width}px';e.style.height='{height}px';}}")
    page.set_viewport_size({"width": width, "height": height})
    el.screenshot(path=png_path, omit_background=True)

# ─────────────────────────────────────────────────────────────
# 4.  Global event bus
# ─────────────────────────────────────────────────────────────
class GlobalEventManager:
    def __init__(self):
        self.events: List[str] = []

    def log_event(self, event: str):
        self.events.append(event)

    def get_summary(self) -> str:
        if not self.events:
            return "No events yet."
        if len(self.events) <= 10:
            return "\n".join(self.events)
        return "\n".join(self.events[:2] + ["\n…older events hidden…\n"] + self.events[-8:])

# ─────────────────────────────────────────────────────────────
# 5.  Basic IO helpers
# ─────────────────────────────────────────────────────────────
def get_abs_path(filepath: str) -> str:
    return os.path.join(WORKSPACE_DIR, filepath.lstrip("/"))

def file_io_tool(action: str, filepath: str, content: str = "") -> str:
    abs_path = get_abs_path(filepath)
    try:
        if action == "save":
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[Success] Saved {filepath}"
        elif action == "load":
            if not os.path.exists(abs_path):
                return f"[Error] File {filepath} does not exist."
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        return f"[Error] File IO: {e}"

def load_json_file(filepath: str, default):
    abs_path = get_abs_path(filepath)
    if not os.path.exists(abs_path):
        return default
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_workspace_file(filepath: str, data: Dict) -> str:
    abs_path = get_abs_path(filepath)
    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"[Success] Saved {filepath}"
    except Exception as e:
        return f"[Error] {e}"

def send_message_tool(source: str, target: str, message: str) -> str:
    inbox_dir = get_abs_path("inbox")
    os.makedirs(inbox_dir, exist_ok=True)
    inbox_file = os.path.join(inbox_dir, f"{target}.md")
    with open(inbox_file, "a", encoding="utf-8") as f:
        f.write(f"### From: {source}\n{message}\n\n---\n\n")
    return f"[Success] Message delivered to {target}."

def python_run_tool(code: str) -> str:
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    env: Dict = {}
    safe = code.replace("unittest.main()", "unittest.main(exit=False)")
    old_cwd = os.getcwd()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    err = ""
    try:
        os.chdir(WORKSPACE_DIR)
        exec(safe, env)
    except Exception as e:
        err = f"[Python Error] {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        os.chdir(old_cwd)
    out = buf.getvalue().strip()
    return f"[Console Output]\n{out}" + (f"\n\n{err}" if err else "")

def validate_agent_file_write(agent_name: str, filepath: str) -> Tuple[bool, str]:
    normalized = filepath.lstrip("/")
    allowed = AGENT_WRITE_RULES.get(agent_name)
    if allowed is None:
        return True, ""
    if normalized in allowed:
        return True, ""
    return False, f"[Error] {agent_name} is not allowed to save '{normalized}'."

# ─────────────────────────────────────────────────────────────
# 6.  SVG utilities  (sanitize, parse, preflight)
# ─────────────────────────────────────────────────────────────
def _parse_rgba_color(value: str) -> Optional[Tuple[str, float]]:
    m = re.fullmatch(
        r"rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([0-9]*\.?[0-9]+)\s*\)",
        (value or "").strip(), re.IGNORECASE)
    if not m:
        return None
    r, g, b = [max(0, min(255, int(m.group(i)))) for i in range(1, 4)]
    a = max(0.0, min(1.0, float(m.group(4))))
    return f"#{r:02X}{g:02X}{b:02X}", a

def sanitize_svg_for_compatibility(svg_code: str) -> str:
    try:
        # ── Pre-parse text fixes ──────────────────────────────────────────
        # 1. xlink:href → href  (Chrome has deprecated xlink:href; also fixes
        #    "unbound prefix" XML parse errors when xmlns:xlink is not declared)
        svg_code = re.sub(r'\sxlink:href=', ' href=', svg_code)
        # 2. Remove the xlink namespace declaration if present (no longer needed)
        svg_code = re.sub(r'\s+xmlns:xlink="[^"]*"', '', svg_code)

        clean = re.sub(r'\sxmlns="[^"]+"', '', svg_code, count=1)
        root  = ET.fromstring(clean)
        if root.tag.split('}')[-1] != "svg":
            return svg_code
        if "viewBox" in root.attrib:
            parts = root.attrib["viewBox"].replace(",", " ").split()
            if len(parts) == 4:
                if "width"  not in root.attrib: root.set("width",  parts[2])
                if "height" not in root.attrib: root.set("height", parts[3])
        for elem in root.iter():
            for attr, oattr in (("fill","fill-opacity"),("stroke","stroke-opacity")):
                parsed = _parse_rgba_color(elem.attrib.get(attr,""))
                if not parsed:
                    continue
                hex_c, alpha = parsed
                elem.set(attr, hex_c)
                try:
                    alpha *= float(elem.attrib.get(oattr, "1"))
                except ValueError:
                    pass
                elem.set(oattr, f"{alpha:.3f}".rstrip("0").rstrip(".") or "0")
        root.set("xmlns", "http://www.w3.org/2000/svg")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
    except Exception:
        return svg_code

def _parse_svg_tree_dict(svg_code: str) -> Dict:
    clean = re.sub(r'\sxmlns="[^"]+"', '', svg_code, count=1)
    root  = ET.fromstring(clean)
    def _node(el, depth=0, idx=0):
        tag  = el.tag.split('}')[-1]
        node = {"tag": tag, "z_order_level": f"{depth}-{idx}"}
        for k in ("id","class","viewBox","cx","cy","r","rx","ry","x","y","width","height",
                  "d","fill","stroke","stroke-width","opacity","transform","clip-path"):
            if k in el.attrib:
                v = el.attrib[k]
                node[k] = v[:120] + "..." if len(v) > 120 else v
        children = list(el)
        node["child_count"] = len(children)
        if children:
            node["children"] = [_node(c, depth+1, i) for i, c in enumerate(children)]
        return node
    return _node(root)

def extract_svg_tree_to_json(svg_code: str) -> str:
    try:
        return json.dumps(_parse_svg_tree_dict(svg_code), indent=2, ensure_ascii=False)
    except Exception as e:
        return f'{{"error": "parse failed: {e}"}}'

def _numbers_from_text(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]

def _bbox_from_svg_element(el) -> Optional[Tuple[float,float,float,float]]:
    tag   = el.tag.split('}')[-1]
    attrs = el.attrib
    def combine(bbs):
        valid = [b for b in bbs if b]
        if not valid: return None
        return (min(b[0] for b in valid), min(b[1] for b in valid),
                max(b[2] for b in valid), max(b[3] for b in valid))
    try:
        if tag == "rect":
            x,y,w,h = float(attrs.get("x",0)),float(attrs.get("y",0)),\
                      float(attrs.get("width",0)),float(attrs.get("height",0))
            return (x,y,x+w,y+h)
        if tag == "circle":
            cx,cy,r = float(attrs.get("cx",0)),float(attrs.get("cy",0)),float(attrs.get("r",0))
            return (cx-r,cy-r,cx+r,cy+r)
        if tag == "ellipse":
            cx,cy=float(attrs.get("cx",0)),float(attrs.get("cy",0))
            rx,ry=float(attrs.get("rx",0)),float(attrs.get("ry",0))
            return (cx-rx,cy-ry,cx+rx,cy+ry)
        if tag in {"line","polyline","polygon"}:
            nums = _numbers_from_text(attrs.get("points",""))
            if len(nums) >= 2:
                return (min(nums[0::2]),min(nums[1::2]),max(nums[0::2]),max(nums[1::2]))
            if tag == "line":
                xs=[float(attrs.get("x1",0)),float(attrs.get("x2",0))]
                ys=[float(attrs.get("y1",0)),float(attrs.get("y2",0))]
                return (min(xs),min(ys),max(xs),max(ys))
        if tag == "path":
            d = attrs.get("d","").strip()
            if not d: return None
            box = parse_path(d).bbox()
            if not box: return None
            xmin,xmax,ymin,ymax = box
            return (xmin,ymin,xmax,ymax)
        if tag == "g":
            return combine([_bbox_from_svg_element(c) for c in list(el)])
    except Exception:
        pass
    return None

def build_svg_element_index(svg_code: str) -> Dict:
    try:
        clean = re.sub(r'\sxmlns="[^"]+"', '', svg_code, count=1)
        root  = ET.fromstring(clean)
    except Exception as e:
        return {"error": str(e), "by_id": {}, "doc_order": {}}
    by_id: Dict = {}
    doc_order: Dict = {}
    counter = [0]
    def walk(el):
        eid = el.attrib.get("id","").strip()
        if eid:
            by_id[eid]      = {"tag": el.tag.split('}')[-1],
                                "bbox": _bbox_from_svg_element(el),
                                "doc_index": counter[0]}
            doc_order[eid]  = counter[0]
        counter[0] += 1
        for c in list(el):
            walk(c)
    walk(root)
    return {"by_id": by_id, "doc_order": doc_order}

def compute_structural_drift(svg_before: str, svg_after: str) -> Dict:
    try:
        ids_before = set(build_svg_element_index(svg_before).get("by_id",{}).keys())
        ids_after  = set(build_svg_element_index(svg_after ).get("by_id",{}).keys())
    except Exception:
        return {"added":[],"removed":[],"drift_score":0,"error":"parse_failed"}
    ids_before.discard(""); ids_after.discard("")
    added   = ids_after  - ids_before
    removed = ids_before - ids_after
    return {"added": list(added), "removed": list(removed),
            "drift_score": len(added) + len(removed)}

def _parse_color_is_whiteish(v: str) -> bool:
    v = (v or "").strip().lower()
    if v in {"white","#fff","#ffffff"}: return True
    m = re.fullmatch(r"#([0-9a-f]{6})", v)
    if not m: return False
    return all(int(m.group(1)[i:i+2],16) >= 238 for i in range(0,6,2))

def svg_static_preflight(svg_code: str, png_path: str) -> Tuple[List[str], List[str]]:
    """Deterministic checks. Returns (blocking_issues, warnings)."""
    blocking: List[str] = []
    warnings: List[str] = []
    try:
        clean = re.sub(r'\sxmlns="[^"]+"', '', svg_code, count=1)
        root  = ET.fromstring(clean)
    except Exception as e:
        return [f"Invalid SVG XML: {e}"], []

    vb = root.attrib.get("viewBox","")
    if vb.replace(","," ").split() != ["0","0","256","256"]:
        blocking.append(f"viewBox must be '0 0 256 256', got '{vb or 'missing'}'.")

    has_contrast_outline = False
    clip_path_stroke_only = False
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        stroke = elem.attrib.get("stroke","")
        try:
            sw = float(elem.attrib.get("stroke-width","0").replace("px",""))
        except ValueError:
            sw = 0
        if sw >= 2 and not _parse_color_is_whiteish(stroke) and stroke.lower() != "none":
            has_contrast_outline = True
        if tag == "clipPath":
            for child in elem.iter():
                ctag = child.tag.split('}')[-1]
                if ctag in {"path","line","polyline"}:
                    if child.attrib.get("stroke") and child.attrib.get("fill","").lower() == "none":
                        clip_path_stroke_only = True

    if clip_path_stroke_only:
        blocking.append("clipPath contains stroke-only fill='none' geometry; clip paths are fill-based.")

    try:
        with Image.open(png_path) as img:
            rgba  = img.convert("RGBA")
            alpha = rgba.getchannel("A")
            bbox  = alpha.getbbox()
            if not bbox:
                blocking.append("Rendered PNG is fully transparent or blank.")
            else:
                w, h    = rgba.size
                l,t,r,b = bbox
                if l<=1 or t<=1 or r>=w-1 or b>=h-1:
                    warnings.append("Rendered icon touches canvas boundary.")
                pixels      = rgba.load()
                opaque = whiteish = 0
                for py in range(t,b):
                    for px in range(l,r):
                        pr,pg,pb,pa = pixels[px,py]
                        if pa <= 32: continue
                        opaque += 1
                        if min(pr,pg,pb) >= 238: whiteish += 1
                if opaque and whiteish/opaque > 0.45 and not has_contrast_outline:
                    blocking.append(
                        ">45% visible pixels near-white without outline — low contrast on white UI.")
    except Exception as e:
        warnings.append(f"Could not inspect PNG pixels: {e}")

    return blocking, warnings

# ─────────────────────────────────────────────────────────────
# 7.  Semantic Tree helpers
# ─────────────────────────────────────────────────────────────
def validate_semantic_tree(tree_text: str) -> Dict:
    """Light validation of the semantic_tree.json produced by TreeBuilder."""
    result = {"valid": False, "errors": [], "warnings": [], "tree_obj": {}}
    try:
        obj = json.loads(tree_text)
    except Exception as e:
        result["errors"].append(f"invalid_json: {e}")
        return result

    for key in ("version","canvas","tree","material_library"):
        if key not in obj:
            result["errors"].append(f"missing_top_level_key: {key}")

    canvas = obj.get("canvas", {})
    vb = canvas.get("viewBox","")
    if vb.replace(","," ").split() != ["0","0","256","256"]:
        result["errors"].append(f"invalid_canvas_viewBox: '{vb}'")

    def check_tree(node, path="root"):
        if not isinstance(node, dict):
            result["errors"].append(f"node_not_dict at {path}")
            return
        if "id" not in node:
            result["errors"].append(f"missing_id at {path}")
        if node.get("is_leaf"):
            if "geometry" not in node:
                result["warnings"].append(f"leaf_missing_geometry: {node.get('id','?')}")
        else:
            for i, child in enumerate(node.get("children", [])):
                check_tree(child, f"{path}.{node.get('id','?')}.children[{i}]")

    check_tree(obj.get("tree", {}))

    # Warn about material_library entries that no leaf node references
    material_library = obj.get("material_library", {})
    if isinstance(material_library, dict):
        def collect_refs(node):
            refs = set()
            if node.get("is_leaf"):
                ref = node.get("geometry", {}).get("material_ref", "")
                if ref:
                    refs.add(ref)
            for child in node.get("children", []):
                refs |= collect_refs(child)
            return refs
        used_refs = collect_refs(obj.get("tree", {}))
        for mat_id in material_library:
            if mat_id not in used_refs:
                result["warnings"].append(
                    f"unused_material: '{mat_id}' defined in material_library "
                    f"but not referenced by any leaf node — add the missing node "
                    f"or remove the material entry")

    result["valid"] = not result["errors"]
    result["tree_obj"] = obj
    return result

# ─────────────────────────────────────────────────────────────
# 8.  Graph Builder  (Python-deterministic + LLM for cross-parent edges)
# ─────────────────────────────────────────────────────────────
CANONICAL_EDGE_TYPES = {"OCCLUDES","CONTAINS","INSIDE_BOUNDS_OF","ATTACHES_TO","ALIGNS_WITH"}

def extract_leaves_from_tree(node: Dict,
                              parent_chain: Optional[List[str]] = None,
                              sibling_index: int = 0) -> List[Dict]:
    parent_chain = parent_chain or []
    if node.get("is_leaf"):
        return [{**node,
                 "parent_chain":   parent_chain,
                 "sibling_index":  sibling_index}]
    leaves = []
    for i, child in enumerate(node.get("children", [])):
        leaves.extend(extract_leaves_from_tree(child, parent_chain + [node["id"]], i))
    return leaves

def _bboxes_overlap(a: Optional[List], b: Optional[List]) -> bool:
    """
    Check whether two bbox_norm [x, y, w, h] rectangles overlap.
    Returns True only when there is actual intersection area (touching edges
    don't count). If either bbox is missing/malformed, conservatively returns
    True so the OCCLUDES edge is still generated (safe default).
    """
    if not (isinstance(a, list) and len(a) == 4 and
            isinstance(b, list) and len(b) == 4):
        return True  # unknown → keep the edge
    try:
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        # no overlap when one rect is entirely to the left/right/above/below the other
        if ax + aw <= bx or bx + bw <= ax:
            return False
        if ay + ah <= by or by + bh <= ay:
            return False
        return True
    except (TypeError, ValueError):
        return True  # malformed → keep the edge

def build_intra_parent_edges(leaves: List[Dict]) -> List[Dict]:
    """
    Sibling order within same parent → deterministic OCCLUDES edges.

    An OCCLUDES edge is only generated when the two siblings' bounding boxes
    actually overlap. Side-by-side peers (e.g. left_eye / right_eye) don't
    overlap, so no edge is created — this prevents spurious ordering constraints
    that confuse the topological sort.
    """
    by_parent: Dict[str, List[Dict]] = defaultdict(list)
    for leaf in leaves:
        if leaf["parent_chain"]:
            by_parent[leaf["parent_chain"][-1]].append(leaf)
    edges = []
    for siblings in by_parent.values():
        siblings.sort(key=lambda x: x.get("sibling_index", 0))
        for i in range(len(siblings) - 1):
            bbox_a = siblings[i  ].get("geometry", {}).get("bbox_norm")
            bbox_b = siblings[i+1].get("geometry", {}).get("bbox_norm")
            if not _bboxes_overlap(bbox_a, bbox_b):
                # non-overlapping siblings — draw order between them is free,
                # no OCCLUDES edge needed
                continue
            # later sibling visually occludes earlier sibling
            edges.append({
                "type":           "OCCLUDES",
                "source":         siblings[i+1]["id"],
                "target":         siblings[i]["id"],
                "inferred_from":  "sibling_order",
            })
    return edges

def topological_sort_draw_order(leaf_ids: List[str],
                                 edges: List[Dict]) -> List[str]:
    """
    Kahn's algorithm deriving draw order from edge constraints.

    Edge semantics → draw-order dependency (target drawn before source):
      OCCLUDES          source covers target   → target first
      INSIDE_BOUNDS_OF  source is inside target → target first (container drawn first,
                        then the element inside it on top)
      CONTAINS          target contains source  → target first (same logic, reversed roles)

    ATTACHES_TO and ALIGNS_WITH carry no draw-order information and are ignored.
    Raises ValueError if a cycle is detected.
    """
    in_degree: Dict[str, int] = defaultdict(int)
    dependents: Dict[str, List[str]] = defaultdict(list)

    for e in edges:
        etype = e.get("type", "")
        src   = e.get("source", "")
        tgt   = e.get("target", "")
    for e in edges:
        etype = e.get("type", "")
        src   = e.get("source", "")
        tgt   = e.get("target", "")
        if etype == "OCCLUDES":
            # "src OCCLUDES tgt" → tgt drawn first, src drawn on top
            # src must wait for tgt → in_degree[src]++
            in_degree[src] += 1
            dependents[tgt].append(src)
        elif etype == "INSIDE_BOUNDS_OF":
            # "src INSIDE_BOUNDS_OF tgt" → src is inside tgt (container)
            # tgt (container) drawn first, src drawn on top inside it
            # src must wait for tgt → in_degree[src]++
            in_degree[src] += 1
            dependents[tgt].append(src)
        elif etype == "CONTAINS":
            # "src CONTAINS tgt" → src is the container, tgt is the content
            # src (container) drawn first, tgt drawn on top inside it
            # tgt must wait for src → in_degree[tgt]++
            in_degree[tgt] += 1
            dependents[src].append(tgt)

    queue  = deque(n for n in leaf_ids if in_degree[n] == 0)
    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(order) != len(leaf_ids):
        cycle_nodes = [n for n in leaf_ids if n not in order]
        raise ValueError(f"Cycle in edges involving: {cycle_nodes}")
    return order

def _llm_generate_cross_parent_edges(leaves: List[Dict],
                                      intra_edges: List[Dict],
                                      material_library: Dict,
                                      canvas: Dict) -> List[Dict]:
    """
    Ask the LLM to add cross-parent topology edges only.
    Returns a (possibly empty) list of new edge dicts.
    """
    leaf_summary = [
        {
            "id":           l["id"],
            "role":         l.get("role",""),
            "parent_chain": l.get("parent_chain",[]),
            "bbox_norm":    l.get("geometry",{}).get("bbox_norm"),
            "importance":   l.get("importance","secondary"),
        }
        for l in leaves
    ]
    # Extract layer/group names from parent_chains to give LLM context about
    # intended foreground/background grouping in the semantic tree
    group_context = {}
    for l in leaves:
        chain = l.get("parent_chain", [])
        if len(chain) >= 2:
            group_id = chain[-1]
            if group_id not in group_context:
                group_context[group_id] = []
            group_context[group_id].append(l["id"])

    prompt = (
        "You are a topology expert for SVG scene graphs.\n\n"
        f"Canvas: {json.dumps(canvas)}\n"
        f"Materials: {json.dumps(material_library)}\n\n"
        "Leaf nodes (geometric primitives):\n"
        f"{json.dumps(leaf_summary, indent=2)}\n\n"
        "Semantic grouping (parent group → child leaf nodes):\n"
        f"{json.dumps(group_context, indent=2)}\n\n"
        "Already-inferred intra-parent OCCLUDES edges:\n"
        f"{json.dumps(intra_edges, indent=2)}\n\n"
        "Task: Add ONLY cross-parent topology edges that are NOT already listed above.\n"
        "Allowed types: OCCLUDES, CONTAINS, INSIDE_BOUNDS_OF, ATTACHES_TO, ALIGNS_WITH.\n"
        "For ALIGNS_WITH, add optional field 'axis': 'x'|'y'|'both'.\n"
        "For ATTACHES_TO, add optional field 'anchor': e.g. 'top_left'.\n"
        "RULES:\n"
        "- Do NOT repeat or contradict existing intra-parent edges.\n"
        "- Do NOT invent nodes that are not in the leaf list.\n"
        "- Prefer fewer, well-justified edges over exhaustive listing.\n"
        "- PAY ATTENTION TO LAYERING INTENT: if the semantic tree has groups named\n"
        "  'front', 'foreground', 'back', 'background', or similar, nodes in the\n"
        "  front group should OCCLUDES nodes in the back group that overlap spatially.\n"
        "  For example, if 'fox_front_layer' contains 'fox_tail' and 'fox_back_layer'\n"
        "  contains 'fox_torso', and the orb sits between them, you should add:\n"
        "    fox_tail OCCLUDES orb_center  (tail is in front of orb)\n"
        "    orb_center OCCLUDES fox_torso  (orb is in front of back layer)\n"
        "  This is the correct way to express depth in a layered scene.\n"
        "- Return ONLY a JSON array of edge objects with keys: type, source, target.\n"
        "- No markdown, no explanation. Raw JSON array only.\n"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return a raw JSON array of edge objects only."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "[]"
        # model might wrap array in {"edges": [...]}
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            edges_raw = parsed
        elif isinstance(parsed, dict):
            edges_raw = parsed.get("edges", [])
        else:
            edges_raw = []

        valid_edges = []
        for e in edges_raw:
            if not isinstance(e, dict): continue
            etype = str(e.get("type","")).strip().upper()
            src   = str(e.get("source","")).strip()
            tgt   = str(e.get("target","")).strip()
            if etype not in CANONICAL_EDGE_TYPES: continue
            leaf_ids = {l["id"] for l in leaves}
            if src not in leaf_ids or tgt not in leaf_ids: continue
            clean_e = {"type": etype, "source": src, "target": tgt}
            if "axis"   in e: clean_e["axis"]   = str(e["axis"])
            if "anchor" in e: clean_e["anchor"] = str(e["anchor"])
            valid_edges.append(clean_e)
        return valid_edges
    except Exception as ex:
        print(f"[GraphBuilder] LLM edge generation failed: {ex}")
        return []

def build_primitive_graph_tool(graph_json_override: str = "") -> str:
    """
    Tool called by SVG_Engineer.
    Reads semantic_tree.json → outputs primitive_graph.json.
    If graph_json_override is provided (Tier-2 retry), validates and saves it directly
    after re-running topological sort (so draw_order stays algorithm-derived).
    """
    if graph_json_override.strip():
        # Tier-2: LLM has revised edge list → re-derive draw_order
        try:
            revised = json.loads(graph_json_override)
        except Exception as e:
            return f"[Error] graph_json_override is not valid JSON: {e}"
        nodes      = revised.get("nodes", [])
        edges      = revised.get("edges", [])
        leaf_ids   = [n["id"] for n in nodes]
        try:
            draw_order = topological_sort_draw_order(leaf_ids, edges)
        except ValueError as e:
            return f"[Error] Cycle in revised graph edges: {e}"
        revised["draw_order"] = draw_order
        file_io_tool("save", PRIMITIVE_GRAPH_FILE,
                     json.dumps(revised, ensure_ascii=False, indent=2))
        return f"[Success] Primitive graph updated from Tier-2 override. draw_order={draw_order}"

    # Normal path: build from semantic tree
    tree_text = file_io_tool("load", SEMANTIC_TREE_FILE)
    if "[Error]" in tree_text:
        return f"[Error] {SEMANTIC_TREE_FILE} missing. TreeBuilder must run first."

    val = validate_semantic_tree(tree_text)
    if not val["valid"]:
        return f"[Error] Semantic tree invalid: {val['errors']}"

    tree_obj         = val["tree_obj"]
    tree_node        = tree_obj["tree"]
    material_library = tree_obj.get("material_library", {})
    canvas           = tree_obj.get("canvas", {})

    # Step A: deterministic leaf extraction + intra-parent edges
    leaves       = extract_leaves_from_tree(tree_node)
    intra_edges  = build_intra_parent_edges(leaves)

    if not leaves:
        return "[Error] Semantic tree has no leaf nodes (is_leaf: true)."

    # Step B: LLM adds cross-parent edges
    cross_edges = _llm_generate_cross_parent_edges(leaves, intra_edges, material_library, canvas)
    all_edges   = intra_edges + cross_edges

    # Step C: topological sort → draw_order
    leaf_ids = [l["id"] for l in leaves]
    try:
        draw_order = topological_sort_draw_order(leaf_ids, all_edges)
    except ValueError as e:
        # Cycle detected; drop the offending cross-parent edges, retry with intra-only
        print(f"[GraphBuilder] Cycle detected ({e}). Falling back to intra-parent edges only.")
        all_edges  = intra_edges
        draw_order = topological_sort_draw_order(leaf_ids, all_edges)

    # Clean leaf nodes for output (drop internal bookkeeping fields)
    clean_leaves = []
    for l in leaves:
        node = {k: v for k, v in l.items()
                if k not in ("parent_chain", "sibling_index")}
        clean_leaves.append(node)

    graph = {
        "canvas":     canvas,
        "nodes":      clean_leaves,
        "edges":      all_edges,
        "draw_order": draw_order,
    }
    file_io_tool("save", PRIMITIVE_GRAPH_FILE,
                 json.dumps(graph, ensure_ascii=False, indent=2))
    return (f"[Success] Primitive graph built: {len(clean_leaves)} nodes, "
            f"{len(all_edges)} edges, draw_order={draw_order}")

def validate_graph_constraints(svg_code: str, graph_text: str) -> Tuple[List[str], List[str]]:
    """Check SVG against primitive_graph constraints. Returns (blocking, warnings)."""
    blocking: List[str] = []
    warnings: List[str] = []
    if not graph_text or "[Error]" in graph_text:
        return blocking, warnings
    try:
        graph = json.loads(graph_text)
    except Exception:
        return blocking, warnings

    svg_index = build_svg_element_index(svg_code)
    if svg_index.get("error"):
        blocking.append(f"Could not parse SVG for constraint check: {svg_index['error']}")
        return blocking, warnings

    by_id      = svg_index["by_id"]
    doc_order  = svg_index["doc_order"]
    nodes      = graph.get("nodes", [])
    edges      = graph.get("edges", [])
    canvas_w   = float(graph.get("canvas",{}).get("viewBox","0 0 256 256").split()[2] or 256)
    canvas_h   = float(graph.get("canvas",{}).get("viewBox","0 0 256 256").split()[3] or 256)

    # Node presence + position drift + shape-specific checks
    for node in nodes:
        nid = str(node.get("id","")).strip()
        if not nid: continue
        if nid not in by_id:
            warnings.append(f"graph_node_missing_in_svg: {nid}")
            continue

        svg_elem = by_id[nid]

        # polygon/polyline with no points attribute → invisible geometry (Tier-1)
        if svg_elem.get("tag") in {"polygon","polyline"}:
            # bbox will be None if points is missing/empty
            if svg_elem.get("bbox") is None:
                blocking.append(
                    f"polygon_missing_points: {nid} — "
                    f"<{svg_elem['tag']}> has no usable 'points' attribute; "
                    f"element renders as invisible. Fix in SVG code (Tier-1).")

        lsg_bbox = node.get("geometry",{}).get("bbox_norm")
        svg_bbox = svg_elem.get("bbox")
        if svg_bbox and isinstance(lsg_bbox, list) and len(lsg_bbox) == 4:
            try:
                nx,ny,nw,nh = [float(v) for v in lsg_bbox]
                exp_cx = (nx + nw/2) * canvas_w
                exp_cy = (ny + nh/2) * canvas_h
                sx1,sy1,sx2,sy2 = svg_bbox
                act_cx = (sx1+sx2)/2; act_cy = (sy1+sy2)/2
                drift = abs(act_cx-exp_cx) + abs(act_cy-exp_cy)
                if drift > 30:
                    warnings.append(
                        f"position_drift: {nid}, expected≈({exp_cx:.0f},{exp_cy:.0f}), "
                        f"actual≈({act_cx:.0f},{act_cy:.0f}), drift={drift:.0f}px")
            except Exception:
                pass

    # Edge constraints
    for edge in edges:
        etype  = str(edge.get("type","")).upper()
        source = str(edge.get("source","")).strip()
        target = str(edge.get("target","")).strip()
        if source not in by_id or target not in by_id: continue
        sb = by_id[source].get("bbox"); tb = by_id[target].get("bbox")
        if not sb or not tb:
            warnings.append(f"bbox_unavailable: {etype}:{source}->{target}")
            continue
        sx1,sy1,sx2,sy2 = sb; tx1,ty1,tx2,ty2 = tb
        if etype in {"CONTAINS","INSIDE_BOUNDS_OF"}:
            if tx1<sx1 or ty1<sy1 or tx2>sx2 or ty2>sy2:
                warnings.append(f"{etype.lower()}_violation: {source}->{target}")
        elif etype == "OCCLUDES":
            if doc_order.get(source,0) <= doc_order.get(target,0):
                warnings.append(f"occludes_doc_order_violation: {source}->{target}")
        elif etype == "ALIGNS_WITH":
            axis = str(edge.get("axis","x")).lower()
            threshold = canvas_w * 0.05
            if axis in {"x","both"}:
                if abs((sx1+sx2)/2 - (tx1+tx2)/2) > threshold:
                    warnings.append(f"aligns_with_x_violation: {source}->{target}")
            if axis in {"y","both"}:
                if abs((sy1+sy2)/2 - (ty1+ty2)/2) > threshold:
                    warnings.append(f"aligns_with_y_violation: {source}->{target}")

    return blocking, warnings

# ─────────────────────────────────────────────────────────────
# 9.  Candidate manifest helpers
# ─────────────────────────────────────────────────────────────
def save_candidate_manifest(files: List[str], expected: int,
                             feedback: str = "", status: str = "ready",
                             validation_results: Optional[List[Dict]] = None):
    manifest = {
        "run_id":          f"svg_batch_{int(time.time())}",
        "generated_at":    int(time.time()),
        "status":          status,
        "expected_count":  expected,
        "candidate_files": files,
        "feedback_retry":  bool(feedback.strip()),
    }
    if validation_results is not None:
        manifest["validation_results"] = validation_results
    file_io_tool("save", CANDIDATE_MANIFEST_FILE,
                 json.dumps(manifest, ensure_ascii=False, indent=2))

def load_candidate_manifest() -> Tuple[List[str], str]:
    raw = file_io_tool("load", CANDIDATE_MANIFEST_FILE)
    if "[Error]" in raw:
        return [], f"manifest_missing"
    try:
        m = json.loads(raw)
    except Exception as e:
        return [], f"manifest_invalid_json: {e}"
    if m.get("status") != "ready":
        return [], f"manifest_not_ready: {m.get('status')}"
    files = [f for f in m.get("candidate_files",[]) if isinstance(f,str)]
    return files, f"manifest_ok: {m.get('run_id','?')}"

# ─────────────────────────────────────────────────────────────
# 10.  SVG Generation  (one-shot and autoregressive)
# ─────────────────────────────────────────────────────────────
_SVG_ENGINEER_SYSTEM = (
    "You are an Elite SVG Engineer. You translate a validated Primitive Graph "
    "into one complete, polished, production-ready SVG icon (viewBox '0 0 256 256'). "
    "You are a code-only actor: output <svg>…</svg> and nothing else unless asked for a <reflection>."
)

_SVG_CHEATSHEET = """
=== SVG ENGINEER CHEAT SHEET ===
1. CANVAS: viewBox='0 0 256 256'. Every coordinate in that space.
2. ID RULE: every SVG element that corresponds to a graph node MUST use the node's exact id.
3. DRAW ORDER: render nodes strictly in the draw_order list (first = bottom layer).
4. CLIPPING: use <clipPath> + <g clip-path="url(#…)"> for CONTAINS/INSIDE_BOUNDS_OF edges.
5. SYMMETRY: define repeated shapes once in <defs>, reuse with <use transform="translate(…)">.
6. ORGANIC FUSION: for rounded/organic subjects, use a background <g> with thick stroke then
   overlay the same shapes with the body fill and stroke='none' (Double-Layering Trick).
7. ALWAYS: stroke-linejoin='round' stroke-linecap='round' on paths.
8. Output ONE valid <svg> block.
=================================
"""

def _extract_svg_from_response(raw: str) -> Optional[str]:
    m = re.search(r"(<svg[\s\S]*?</svg>)", raw, re.DOTALL)
    return m.group(1).strip() if m else None

def _generate_one_svg(prompt_context: str, temperature: float) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SVG_ENGINEER_SYSTEM},
                {"role": "user",   "content": prompt_context},
            ],
            temperature=temperature,
        )
        raw = resp.choices[0].message.content or ""
        return _extract_svg_from_response(raw)
    except Exception as e:
        print(f"   [SVG_Engineer] API error: {e}")
        return None

def _build_oneshot_prompt(graph: Dict, user_task: str,
                          feedback: str = "", previous_svg: str = "") -> str:
    graph_text  = json.dumps(graph, indent=2, ensure_ascii=False)
    draw_order  = graph.get("draw_order",[])
    prompt = (
        f"=== PRIMITIVE GRAPH ===\n{graph_text}\n\n"
        f"Draw order (bottom to top): {draw_order}\n\n"
        f"=== USER TASK ===\n{user_task}\n\n"
        f"{_SVG_CHEATSHEET}\n"
        "Generate ONE complete production-ready SVG following the primitive graph exactly.\n"
    )
    if feedback.strip() and previous_svg.strip():
        prev_tree = extract_svg_tree_to_json(previous_svg)
        prompt += (
            f"\n=== PREVIOUS SVG ===\n{previous_svg}\n\n"
            f"=== PREVIOUS SVG TREE ===\n{prev_tree}\n\n"
            f"=== CRITIC FEEDBACK ===\n{feedback}\n\n"
            "MODE: MINIMAL PATCH. Preserve global composition; fix only what feedback mentions.\n"
            "Output <reflection> first, then your SVG.\n"
        )
    return prompt

def _build_autoregressive_prompt(node: Dict, already_drawn: str,
                                  graph: Dict, user_task: str,
                                  material_library: Dict) -> str:
    mat_ref  = node.get("material_ref","")
    material = material_library.get(mat_ref, {})
    return (
        f"=== FULL PRIMITIVE GRAPH (reference) ===\n"
        f"{json.dumps(graph, indent=2, ensure_ascii=False)}\n\n"
        f"=== USER TASK ===\n{user_task}\n\n"
        f"=== ALREADY DRAWN SVG (partial) ===\n{already_drawn or '(nothing yet)'}\n\n"
        f"=== YOUR CURRENT NODE ===\n{json.dumps(node, indent=2, ensure_ascii=False)}\n"
        f"Material: {json.dumps(material)}\n\n"
        "Output ONLY the SVG element(s) for this single node (no <svg> wrapper, no other nodes).\n"
        "Use the node's exact id. Coordinates must be in 0-256 space.\n"
    )

def _svg_gen_one_shot(graph: Dict, user_task: str, k: int,
                      feedback: str, previous_svg: str,
                      graph_text: str) -> List[str]:
    prompt   = _build_oneshot_prompt(graph, user_task, feedback, previous_svg)
    saved: List[str] = []
    debug_dir = os.path.abspath(".debug_logs/svg_candidates")
    os.makedirs(debug_dir, exist_ok=True)
    for i in range(k):
        temp = (0.05 + i*0.05) if feedback.strip() else (0.2 + i*0.15)
        svg  = _generate_one_svg(prompt, temp)
        if svg:
            svg   = sanitize_svg_for_compatibility(svg)
            fname = f"candidate_{i+1}.svg"
            file_io_tool("save", fname, svg)
            _publish_svg(fname, svg)
            saved.append(fname)
        else:
            print(f"   [SVG_Engineer] No SVG in candidate {i+1}")
    return saved

def _svg_gen_autoregressive(graph: Dict, user_task: str, k: int,
                             feedback: str, previous_svg: str,
                             graph_text: str) -> List[str]:
    """Generate k candidates; each built node-by-node along draw_order."""
    draw_order       = graph.get("draw_order",[])
    nodes_by_id      = {n["id"]: n for n in graph.get("nodes",[])}
    material_library = json.loads(file_io_tool("load", SEMANTIC_TREE_FILE) or "{}").get(
                            "material_library", {})
    canvas_vb = graph.get("canvas",{}).get("viewBox","0 0 256 256")
    debug_dir = os.path.abspath(".debug_logs/svg_candidates")
    os.makedirs(debug_dir, exist_ok=True)
    saved: List[str] = []

    # Feedback from Critic is prepended to the system prompt so every node
    # generation is aware of what went wrong in the previous attempt.
    feedback_prefix = ""
    if feedback.strip():
        feedback_prefix = (
            f"=== CRITIC FEEDBACK FROM PREVIOUS ATTEMPT ===\n{feedback.strip()}\n"
            "Apply this feedback when drawing each node below.\n"
            "==============================================\n\n"
        )

    for i in range(k):
        assembled_elements: List[str] = []
        for nid in draw_order:
            node   = nodes_by_id.get(nid)
            if not node: continue
            prompt = _build_autoregressive_prompt(
                node, "\n".join(assembled_elements), graph, user_task, material_library)
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system",
                         "content": feedback_prefix +
                                    "You output only SVG element(s) for a single node. No wrapper."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1 + i*0.05,
                )
                fragment = resp.choices[0].message.content or ""
                # strip any accidental <svg> wrappers the model might add
                fragment = re.sub(r"</?svg[^>]*>","", fragment).strip()
                assembled_elements.append(fragment)
            except Exception as e:
                print(f"   [AR] node {nid} failed: {e}")

        full_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{canvas_vb}" '
            f'width="256" height="256">\n'
            + "\n".join(assembled_elements)
            + "\n</svg>"
        )
        full_svg = sanitize_svg_for_compatibility(full_svg)
        fname    = f"candidate_{i+1}.svg"
        file_io_tool("save", fname, full_svg)
        _publish_svg(fname, full_svg)
        saved.append(fname)
    return saved

def svg_gen_batch_tool(k: int = 3, feedback: str = "",
                       mode: str = "one_shot") -> str:
    """
    Main SVG generation tool.
    Requires primitive_graph.json to exist in workspace.
    mode: 'one_shot' | 'autoregressive'
    """
    k = min(max(k, 1), 3)
    user_task = file_io_tool("load", "user_task.txt")
    if "[Error]" in user_task:
        return "[Error] user_task.txt not found."

    graph_text = file_io_tool("load", PRIMITIVE_GRAPH_FILE)
    if "[Error]" in graph_text:
        return (f"[Error] {PRIMITIVE_GRAPH_FILE} missing. "
                "Call build_primitive_graph first.")
    try:
        graph = json.loads(graph_text)
    except Exception as e:
        return f"[Error] {PRIMITIVE_GRAPH_FILE} is not valid JSON: {e}"

    if not graph.get("draw_order"):
        return f"[Error] {PRIMITIVE_GRAPH_FILE} has no draw_order. Rebuild graph."

    if feedback.strip():
        previous_svg = file_io_tool("load", "current_best_icon.svg")
        if "[Error]" in previous_svg:
            previous_svg = ""
    else:
        previous_svg = file_io_tool("load", "current_best_icon.svg")
        if "[Error]" in previous_svg:
            previous_svg = ""

    # When retrying with Critic feedback, always use one_shot regardless of the
    # global generation_mode setting.  Autoregressive mode accumulates errors
    # across nodes and cannot apply holistic patch instructions from Critic.
    # one_shot has full context of the previous SVG and can make targeted edits.
    effective_mode = mode
    if feedback.strip() and mode == "autoregressive":
        effective_mode = "one_shot"
        print("   [SVG_Engineer] Retry detected — switching to one_shot for feedback-guided patch.")

    save_candidate_manifest([], expected=k, feedback=feedback,
                             status="generating")
    print(f"🎨 [SVG_Engineer] Generating {k} candidates (mode={effective_mode})...")

    if effective_mode == "autoregressive":
        saved_files = _svg_gen_autoregressive(
            graph, user_task, k, feedback, previous_svg, graph_text)
    else:
        saved_files = _svg_gen_one_shot(
            graph, user_task, k, feedback, previous_svg, graph_text)

    if not saved_files:
        save_candidate_manifest([], expected=k, feedback=feedback,
                                 status="failed",
                                 validation_results=[{"valid":False,"errors":["no_candidates"]}])
        return "[Error] Failed to generate any SVG candidates."

    is_retry = bool(feedback.strip())
    # Drift is always measured against current_best_icon.svg.
    drift_reference = ""
    if is_retry:
        raw = file_io_tool("load", "current_best_icon.svg")
        drift_reference = "" if "[Error]" in raw else raw

    validation_results = [
        _validate_svg_candidate(f, user_task, graph_text,
                                 drift_reference if is_retry else "")
        for f in saved_files
    ]
    valid_files   = [r["file"] for r in validation_results if r["valid"]]
    invalid_files = [r for r in validation_results if not r["valid"]]

    if not valid_files:
        save_candidate_manifest([], expected=len(saved_files), feedback=feedback,
                                 status="failed", validation_results=validation_results)
        return (f"[Error] All candidates failed preflight. "
                f"Details: {json.dumps(validation_results, ensure_ascii=False)}")

    save_candidate_manifest(valid_files, expected=len(valid_files),
                             feedback=feedback, validation_results=validation_results)

    summary = f"Preflight: {len(valid_files)}/{len(saved_files)} valid"
    if invalid_files:
        summary += f"; rejected={json.dumps(invalid_files, ensure_ascii=False)}"
    return (f"[Success] Generated {len(saved_files)} candidates (mode={mode}). "
            f"{summary}. Files: {', '.join(valid_files)}. Manifest: {CANDIDATE_MANIFEST_FILE}")

def _validate_svg_candidate(fname: str, user_task: str,
                              graph_text: str, previous_svg: str = "") -> Dict:
    result: Dict = {"file": fname, "valid": False, "errors": [], "warnings": []}
    svg_path = get_abs_path(fname)
    if not os.path.exists(svg_path):
        result["errors"].append("file_missing"); return result
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_code = f.read()
    except Exception as e:
        result["errors"].append(f"read_failed: {e}"); return result
    if not svg_code.strip():
        result["errors"].append("empty_svg"); return result

    # Structural drift check (retry mode only)
    if previous_svg:
        drift = compute_structural_drift(previous_svg, svg_code)
        result["drift_info"] = drift
        if drift.get("drift_score",0) > 5:
            result["errors"].append(
                f"structural_drift_too_high: score={drift['drift_score']} (minimal patch violated)")

    # XML validity + viewBox
    try:
        clean = re.sub(r'\sxmlns="[^"]+"', '', svg_code, count=1)
        root  = ET.fromstring(clean)
    except Exception as e:
        result["errors"].append(f"invalid_xml: {e}"); return result
    if root.tag.split('}')[-1] != "svg":
        result["errors"].append("root_not_svg")
    vb = root.attrib.get("viewBox","")
    if vb.replace(","," ").split() != ["0","0","256","256"]:
        result["errors"].append(f"invalid_viewBox: {vb or 'missing'}")

    # Render + pixel preflight
    if not result["errors"]:
        png_path = svg_path.replace(".svg",".png")
        try:
            svg_to_png_with_browser(svg_path, png_path, 256, 256)
            result["preflight_png"] = os.path.basename(png_path)
            b_issues, warns = svg_static_preflight(svg_code, png_path)
            # Also check graph constraints
            gb_issues, gb_warns = validate_graph_constraints(svg_code, graph_text)
            result["errors"].extend(f"preflight: {x}" for x in b_issues)
            result["errors"].extend(f"graph_constraint: {x}" for x in gb_issues)
            result["warnings"].extend(warns + gb_warns)
        except Exception as e:
            result["errors"].append(f"render_failed: {e}")

    result["valid"] = not result["errors"]
    return result

# ─────────────────────────────────────────────────────────────
# 11. Publish helpers
# ─────────────────────────────────────────────────────────────
def _publish_svg(filename: str, svg_text: str):
    os.makedirs(PUBLISHED_DIR, exist_ok=True)
    dst = os.path.join(PUBLISHED_DIR, os.path.basename(filename))
    with open(dst, "w", encoding="utf-8") as f:
        f.write(svg_text)

def publish_workspace_artifact(filename: str) -> str:
    src = get_abs_path(filename)
    if not os.path.exists(src):
        return f"[Skipped] {filename} not found."
    os.makedirs(PUBLISHED_DIR, exist_ok=True)
    dst = os.path.join(PUBLISHED_DIR, os.path.basename(filename))
    shutil.copy2(src, dst)
    return f"[Success] Published {filename}"

def resolve_final_candidate(preferred: str = "") -> str:
    for name in [preferred]:
        if name and os.path.exists(get_abs_path(name)):
            return name
    manifest_files, _ = load_candidate_manifest()
    for name in manifest_files:
        if os.path.exists(get_abs_path(name)):
            return name
    for fallback in ["current_best_icon.svg", "icons.svg"]:
        if os.path.exists(get_abs_path(fallback)):
            return fallback
    return ""

def finalize_candidate_artifacts(candidate_file: str,
                                   svg_tree_json: str = "",
                                   score_payload: Optional[Dict] = None) -> str:
    resolved = resolve_final_candidate(candidate_file)
    if not resolved:
        return "[Error] No candidate SVG to finalize."
    svg_text = file_io_tool("load", resolved)
    if "[Error]" in svg_text:
        return f"[Error] Failed to load {resolved}"
    clean = sanitize_svg_for_compatibility(svg_text)
    file_io_tool("save", "current_best_icon.svg", clean)
    file_io_tool("save", "icons.svg", clean)
    publish_workspace_artifact("current_best_icon.svg")
    publish_workspace_artifact("icons.svg")
    try:
        svg_to_png_with_browser(get_abs_path("current_best_icon.svg"),
                                get_abs_path("current_best_icon.png"), 256, 256)
        publish_workspace_artifact("current_best_icon.png")
    except Exception as e:
        return f"[Error] SVG finalized but PNG render failed: {e}"
    tree = svg_tree_json or extract_svg_tree_to_json(clean)
    file_io_tool("save", BEST_SVG_TREE_FILE, tree)
    if score_payload:
        save_json_workspace_file(BEST_SCORE_FILE, score_payload)
    return "[Success] Finalized: current_best_icon.svg / .png, icons.svg"

# ─────────────────────────────────────────────────────────────
# 12. Scorer service client
# ─────────────────────────────────────────────────────────────
def compute_metric_scores(png_path: str, prompt: str) -> Dict:
    """Call the external scorer service (CLIP + HPSv2). Degrades gracefully."""
    try:
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = _requests.post(SCORER_URL,
                              json={"image_b64": b64, "prompt": prompt},
                              timeout=300)
        data = resp.json()
        return {"clip": float(data.get("clip",0.0)),
                "hps":  float(data.get("hps", 0.0))}
    except Exception as e:
        print(f"[Warning] Scorer service unavailable: {e}")
        return {"clip": 0.0, "hps": 0.0}

# ─────────────────────────────────────────────────────────────
# 13. Retry counter
# ─────────────────────────────────────────────────────────────
class RetryCounter:
    """Persistent retry counter stored in workspace."""
    def __init__(self):
        data = load_json_file(FAIL_COUNTER_FILE, {"count": 0})
        self.count: int = data.get("count", 0)

    def can_retry(self) -> bool:
        return self.count < MAX_PATCH_RETRIES

    def consume(self):
        self.count += 1
        save_json_workspace_file(FAIL_COUNTER_FILE, {"count": self.count})

    def reset(self):
        self.count = 0
        save_json_workspace_file(FAIL_COUNTER_FILE, {"count": 0})

    def status(self) -> str:
        return f"{self.count}/{MAX_PATCH_RETRIES}"

# ─────────────────────────────────────────────────────────────
# 14. Failure diagnosis  (Tier routing)
# ─────────────────────────────────────────────────────────────
# 14. Failure diagnosis  (Tier routing)
# ─────────────────────────────────────────────────────────────
def diagnose_failure_tier(clip_score: float,
                           graph_violations: List[str],
                           structural_drift_score: int,
                           preflight_blocking: Optional[List[str]] = None,
                           llm_blocking: Optional[List[str]] = None) -> int:
    """
    Strict Tier-1 / Tier-2 separation.

    Tier 1 — SVG code-level defects (send feedback to SVG_Engineer):
      These are implementation bugs that don't require touching the graph.
      Examples: wrong coordinate causing an element to exceed bounds,
      clipPath used incorrectly, visual gaps between connected shapes,
      color/stroke issues, minor misalignment.
      Signals:
        - preflight_blocking has issues (pixel/XML level)
        - LLM flags visual/polish issues without structural mismatch
        - drift is low (SVG_Engineer faithfully followed the graph)
        - CLIP score is in acceptable range (>0.20, semantic structure OK)

    Tier 2 — Graph/semantic structure defects (send feedback to TreeBuilder):
      These require rebuilding semantic_tree.json because the graph itself
      is wrong — missing nodes, wrong occlusion topology, fundamentally
      misplaced elements.
      Signals:
        - graph constraint violations (OCCLUDES/CONTAINS violated in SVG)
        - CLIP severely low (<=0.20): semantic structure is wrong, not just
          rendering (e.g. "cat is missing a leg", "head and body reversed")
        - structural drift >= 3: SVG_Engineer invented structure not in graph,
          suggesting the graph description was too vague or wrong

    Returns 1 or 2.
    """
    preflight_blocking = preflight_blocking or []
    llm_blocking       = llm_blocking or []

    # Hard Tier-2 triggers — graph structure is fundamentally wrong.
    # polygon_missing_points is explicitly excluded: it means SVG_Engineer
    # generated a <polygon> without a points attribute — that is a code
    # generation bug fixable by Tier-1 patch, not a graph structure error.
    TIER2_VIOLATION_PREFIXES = (
        "structural_violation_promoted",
        "occludes_doc_order_violation",
        "contains_violation",
    )
    structural_graph_violations = [
        v for v in graph_violations
        if any(v.startswith(p) for p in TIER2_VIOLATION_PREFIXES)
    ]

    if structural_graph_violations:
        return 2
    if clip_score <= 0.20:
        return 2
    if structural_drift_score >= 3:
        return 2

    # Everything else is Tier-1: SVG code can be patched without touching graph
    return 1

# ─────────────────────────────────────────────────────────────
# 15. TextGrad optimizers
# ─────────────────────────────────────────────────────────────
def textgrad_tier1_patch(svg_code: str, user_task: str,
                          clip_score: float, hps_score: float,
                          preflight_issues: List[str]) -> str:
    """Optimise SVG code directly. Returns patched SVG string."""
    if not _TEXTGRAD_AVAILABLE:
        return svg_code
    try:
        svg_var = tg.Variable(
            svg_code,
            requires_grad=True,
            role_description="SVG icon source code to be improved",
        )
        loss_instruction = (
            f"Task: {user_task}\n"
            f"CLIP text-image alignment: {clip_score:.3f} (target > 0.25)\n"
            f"HPSv2 human preference:    {hps_score:.3f}  (target > 0.28)\n"
            f"Static preflight issues:   {json.dumps(preflight_issues)}\n\n"
            "Evaluate this SVG icon. Provide SPECIFIC, ACTIONABLE gradient feedback:\n"
            "• Which element IDs need color/fill changes and to what values\n"
            "• Which coordinates need adjustment for better visual alignment\n"
            "• What is causing the CLIP score to be low (missing semantic elements?)\n"
            "• What is causing the HPS score to be low (aesthetic issues?)\n"
            "Do NOT suggest adding or removing nodes — that is a structural concern.\n"
        )
        loss_fn = tg.TextLoss(loss_instruction)
        loss    = loss_fn(svg_var)
        loss.backward()
        tg.TGD(parameters=[svg_var]).step()
        result = svg_var.value
        # Ensure we still have valid SVG after TGD
        if _extract_svg_from_response(result):
            return result
        return svg_code   # TGD mangled the SVG; keep original
    except Exception as e:
        print(f"[TextGrad Tier1] Error: {e}")
        return svg_code

def textgrad_tier2_rebuild(graph_json: str, user_task: str,
                            clip_score: float,
                            graph_violations: List[str],
                            structural_drift: Dict) -> str:
    """Optimise primitive_graph JSON. Returns revised graph JSON string."""
    if not _TEXTGRAD_AVAILABLE:
        return graph_json
    try:
        graph_var = tg.Variable(
            graph_json,
            requires_grad=True,
            role_description="primitive graph JSON defining SVG structure and topology",
        )
        loss_instruction = (
            f"Task: {user_task}\n"
            f"CLIP score: {clip_score:.3f} — structural/semantic failure\n"
            f"Graph constraint violations: {json.dumps(graph_violations)}\n"
            f"Structural drift vs graph: {json.dumps(structural_drift)}\n\n"
            "The SVG has structural failures that require fixing the underlying graph.\n"
            "Provide SPECIFIC feedback on:\n"
            "• Which OCCLUDES edges have wrong direction (causing layering failures)\n"
            "• Which leaf nodes are missing (causing CLIP semantic gap)\n"
            "• Which ATTACHES_TO anchor coordinates are wrong\n"
            "• Which bbox_norm values violate containment constraints\n"
            "Guide revision of graph JSON edges and node definitions only.\n"
            "Do NOT suggest SVG code changes.\n"
        )
        loss_fn = tg.TextLoss(loss_instruction)
        loss    = loss_fn(graph_var)
        loss.backward()
        tg.TGD(parameters=[graph_var]).step()
        revised = graph_var.value
        # Validate it's still parseable JSON
        json.loads(revised)
        return revised
    except Exception as e:
        print(f"[TextGrad Tier2] Error: {e}")
        return graph_json

# ─────────────────────────────────────────────────────────────
# 16a. Critic debug reporter
# ─────────────────────────────────────────────────────────────
PROMPT_BASELINES: Dict[str, str] = {}

def _load_prompt_overrides() -> Dict:
    raw = file_io_tool("load", PROMPT_OVERRIDES_FILE)
    if "[Error]" in raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_prompt_overrides(overrides: Dict) -> None:
    file_io_tool("save", PROMPT_OVERRIDES_FILE,
                 json.dumps(overrides, ensure_ascii=False, indent=2))

def _delete_workspace_file(filename: str) -> None:
    path = get_abs_path(filename)
    if os.path.exists(path):
        os.remove(path)

def reset_prompt_overrides_for_task() -> None:
    for filename in (
        PROMPT_OVERRIDES_FILE,
        PROMPT_OVERRIDES_PREV_FILE,
        PROMPT_OVERRIDE_METRIC_FILE,
    ):
        _delete_workspace_file(filename)

def maybe_rollback_prompt_overrides(metric_score: float) -> str:
    metric_data = load_json_file(PROMPT_OVERRIDE_METRIC_FILE, {})
    baseline = metric_data.get("metric_score")
    if not isinstance(baseline, (int, float)):
        return ""
    if metric_score >= float(baseline):
        return ""

    prev_raw = file_io_tool("load", PROMPT_OVERRIDES_PREV_FILE)
    if "[Error]" in prev_raw:
        _delete_workspace_file(PROMPT_OVERRIDES_FILE)
        _delete_workspace_file(PROMPT_OVERRIDE_METRIC_FILE)
        return (
            f"Rolled back prompt override: metric {metric_score:.4f} < "
            f"{float(baseline):.4f}; restored empty override."
        )

    file_io_tool("save", PROMPT_OVERRIDES_FILE, prev_raw)
    _delete_workspace_file(PROMPT_OVERRIDE_METRIC_FILE)
    return (
        f"Rolled back prompt override: metric {metric_score:.4f} < "
        f"{float(baseline):.4f}; restored previous override."
    )

def _extract_json_object(text: str) -> Optional[Dict]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _normalize_prompt_overrides(obj: Dict) -> Dict:
    clean: Dict = {}
    for agent in ("TreeBuilder", "SVG_Engineer"):
        block = obj.get(agent)
        if not isinstance(block, dict):
            continue
        out: Dict[str, str] = {}
        role_desc = block.get("role_desc")
        if isinstance(role_desc, str) and role_desc.strip():
            out["role_desc"] = role_desc.strip()
        if out:
            clean[agent] = out
    rationale = obj.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        clean["rationale"] = rationale.strip()
    return clean

def _runtime_role_desc(agent_name: str, base_role_desc: str) -> str:
    override = _load_prompt_overrides().get(agent_name, {}).get("role_desc", "")
    if not override:
        return base_role_desc
    return (
        "BASE ROLE CONTRACT (must remain true; preserve tools, routing, schemas, "
        "and file permissions):\n"
        f"{base_role_desc}\n\n"
        "TEXTGRAD RUNTIME ROLE REWRITE (active strategy for this run):\n"
        f"{override}"
    )

def _runtime_critic_sys_msg(base_sys_msg: str) -> str:
    return base_sys_msg

def textgrad_prompt_rewrite(user_task: str,
                            critic_sys_msg: str,
                            clip_score: float,
                            hps_score: float,
                            tier: int,
                            blocking_issues: List[str],
                            graph_warnings: List[str],
                            preflight_warnings: List[str],
                            critic_advice: str,
                            structural_drift: Dict) -> str:
    """Optimise runtime prompts only. Does not edit tree, graph, or SVG."""
    if not _TEXTGRAD_AVAILABLE:
        return ""
    try:
        seed = json.dumps(_load_prompt_overrides() or {
            "TreeBuilder": {"role_desc": ""},
            "SVG_Engineer": {"role_desc": ""},
            "rationale": "",
        }, ensure_ascii=False, indent=2)
        prompt_var = tg.Variable(
            seed,
            requires_grad=True,
            role_description="runtime prompt override JSON for SVG multi-agent pipeline",
        )
        loss_instruction = (
            f"Task: {user_task}\n"
            f"CLIP text-image alignment: {clip_score:.3f} (target > 0.25)\n"
            f"HPSv2 human preference:    {hps_score:.3f}  (target > 0.28)\n"
            f"Failure tier: {tier}\n"
            f"Blocking issues: {json.dumps(blocking_issues, ensure_ascii=False)}\n"
            f"Graph warnings: {json.dumps(graph_warnings, ensure_ascii=False)}\n"
            f"Preflight warnings: {json.dumps(preflight_warnings, ensure_ascii=False)}\n"
            f"Structural drift: {json.dumps(structural_drift, ensure_ascii=False)}\n"
            f"Critic advice: {critic_advice}\n\n"
            "Rewrite the runtime prompts to improve the next attempt. You may "
            "substantially rewrite role_desc for TreeBuilder and SVG_Engineer. "
            "Do not rewrite Critic. Keep the original contracts as "
            "anchors: do not change tool schemas, routing responsibilities, JSON "
            "output schema, file permissions, or directly edit semantic_tree, "
            "primitive_graph, or SVG artifacts. Output ONLY valid JSON with keys "
            "TreeBuilder.role_desc, SVG_Engineer.role_desc, and rationale.\n\n"
            "=== BASE ROLE DESCRIPTIONS ===\n"
            f"{json.dumps(PROMPT_BASELINES, ensure_ascii=False, indent=2)}\n\n"
            "=== BASE CRITIC SYS_MSG ===\n"
            f"{critic_sys_msg}\n"
        )
        loss_fn = (tg.TextLoss(loss_instruction, engine=TEXTGRAD_FORWARD_ENGINE)
                   if TEXTGRAD_FORWARD_ENGINE is not None
                   else tg.TextLoss(loss_instruction))
        loss = loss_fn(prompt_var)
        loss.backward()
        tg.TGD(parameters=[prompt_var]).step()
        parsed = _extract_json_object(prompt_var.value)
        if not parsed:
            return "TGD prompt rewrite failed: invalid JSON; kept existing prompts."
        clean = _normalize_prompt_overrides(parsed)
        if not clean:
            return "TGD prompt rewrite produced no usable overrides."
        current_raw = file_io_tool("load", PROMPT_OVERRIDES_FILE)
        if "[Error]" in current_raw:
            current_raw = "{}"
        file_io_tool("save", PROMPT_OVERRIDES_PREV_FILE, current_raw)
        _save_prompt_overrides(clean)
        changed = ", ".join(k for k in clean.keys() if k != "rationale")
        return f"TGD runtime prompt rewrite saved to {PROMPT_OVERRIDES_FILE}: {changed}"
    except Exception as e:
        print(f"[TextGrad PromptRewrite] Error: {e}")
        return f"TGD prompt rewrite error: {e}"

CRITIC_DEBUG_DIR = os.path.abspath("critic_debug")

def _save_critic_debug_report(
    result_payload:     Dict,
    best:               Dict,
    all_candidates:     List[Dict],
    preflight_blocking: List[str],
    preflight_warnings: List[str],
    graph_blocking:     List[str],
    graph_warnings:     List[str],
    llm_result_raw:     str,
    sys_msg_sent:       str,
    reward_ablation:    str,
    use_textgrad:       bool,
) -> None:
    """
    Write a human-readable Markdown report + copy candidate PNGs
    into critic_debug/<timestamp>/ so every Critic run is inspectable.

    Layout of each run directory:
      critic_debug/
        run_<timestamp>/
          report.md          ← full structured report (open in any MD viewer)
          candidate_1.png    ← rendered PNG of each candidate
          candidate_2.png
          candidate_3.png
          best_candidate.png ← copy of the selected winner
    """
    ts       = int(time.time())
    run_dir  = os.path.join(CRITIC_DEBUG_DIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    verdict       = result_payload.get("verdict","?")
    overall_score = result_payload.get("overall_svg_score","?")
    clip_score    = result_payload.get("clip_score")
    hps_score     = result_payload.get("hps_score")
    tier          = result_payload.get("tier")
    retry_status  = result_payload.get("retry_counter","?")
    tgd_result    = result_payload.get("tgd_result","")

    # ── Copy PNGs into run_dir ────────────────────────────────
    for c in all_candidates:
        src = c.get("png_path","")
        if src and os.path.exists(src):
            fname = os.path.basename(src)
            shutil.copy2(src, os.path.join(run_dir, fname))

    best_png = best.get("png_path","")
    if best_png and os.path.exists(best_png):
        shutil.copy2(best_png, os.path.join(run_dir, "best_candidate.png"))

    # ── Build report.md ──────────────────────────────────────
    def _fmt_list(items: List[str], empty: str = "none") -> str:
        return "\n".join(f"- {x}" for x in items) if items else f"_{empty}_"

    def _score_bar(score, max_score=10) -> str:
        """Visual bar so you can scan scores at a glance."""
        if not isinstance(score, (int, float)):
            return "n/a"
        filled = round(score / max_score * 10)
        return "█" * filled + "░" * (10 - filled) + f"  {score}/{max_score}"

    report_lines = [
        f"# Critic Debug Report — run_{ts}",
        "",
        f"**Verdict:** `{verdict}`  |  **Retry:** {retry_status}  |  "
        f"**Ablation:** {reward_ablation}  |  **TextGrad:** {use_textgrad}",
        "",

        "---",
        "## 1. Candidates evaluated",
        "",
    ]
    for c in all_candidates:
        marker = " ← **selected**" if c["file"] == best["file"] else ""
        clip   = f"{c['clip']:.3f}" if c.get("clip") is not None else "n/a"
        hps    = f"{c['hps']:.3f}"  if c.get("hps")  is not None else "n/a"
        report_lines.append(
            f"- `{c['file']}`{marker}  "
            f"CLIP={clip}  HPS={hps}  "
            f"PNG: `{os.path.basename(c.get('png_path','?'))}`"
        )

    report_lines += [
        "",
        "---",
        "## 2. Metric scores (Signal 1)",
        "",
        f"| Metric | Score | Threshold | Status |",
        f"|--------|-------|-----------|--------|",
        f"| CLIP   | {clip_score if clip_score is not None else 'disabled'} "
        f"| > 0.25 | {'⚠️ low' if isinstance(clip_score,float) and clip_score < 0.25 else '✓'} |",
        f"| HPSv2  | {hps_score if hps_score is not None else 'disabled'} "
        f"| > 0.28 | {'⚠️ low' if isinstance(hps_score,float) and hps_score < 0.28 else '✓'} |",
        "",
        "> CLIP measures text-image alignment.  "
        "HPSv2 measures human aesthetic preference.",
        "",

        "---",
        "## 3. Deterministic checks (Signal 2)",
        "",
        "### Preflight blocking issues",
        _fmt_list(preflight_blocking),
        "",
        "### Preflight warnings",
        _fmt_list(preflight_warnings),
        "",
        "### Graph constraint blocking issues",
        _fmt_list(graph_blocking),
        "",
        "### Graph constraint warnings",
        _fmt_list(graph_warnings),
        "",
        "> Blocking issues → automatic FAIL regardless of LLM score.",
        "",

        "---",
        "## 4. LLM vision scoring (Signal 3)",
        "",
        f"**Overall score:** {_score_bar(overall_score)}",
        "",
    ]

    # Per-dimension scores if available in raw response
    try:
        llm_parsed = json.loads(llm_result_raw) if llm_result_raw else {}
        scores = llm_parsed.get("scores", {})
        if scores:
            report_lines.append("| Dimension | Score |")
            report_lines.append("|-----------|-------|")
            for dim, val in scores.items():
                report_lines.append(f"| {dim} | {_score_bar(val, 5)} |")
            report_lines.append("")
    except Exception:
        pass

    report_lines += [
        "### LLM blocking issues",
        _fmt_list(result_payload.get("blocking_issues", [])),
        "",
        "### Aggregated advice",
        result_payload.get("advices","") or "_none_",
        "",
        "### Full raw LLM response",
        "```json",
        llm_result_raw or "(empty)",
        "```",
        "",

        "---",
        "## 5. Pass gate",
        "",
        f"- pass_ready (LLM): `{result_payload.get('pass_ready')}`",
        f"- metric_pass (CLIP/HPS floors): `{result_payload.get('metric_pass')}`",
        f"- overall_score >= 8: `{isinstance(overall_score,int) and overall_score >= 8}`",
        f"- blocking_issues empty: `{not result_payload.get('blocking_issues')}`",
        f"- **Final verdict: `{verdict}`**",
        "",

        "---",
        "## 6. Failure routing (if FAIL)",
        "",
    ]

    if verdict == "FAIL":
        report_lines += [
            f"- Tier: `{tier}`  "
            f"({'SVG-level patch' if tier==1 else 'Graph rebuild'})",
            f"- Drift score: `{result_payload.get('tier')}`",
            f"- TGD result: {tgd_result or '_TextGrad not run_'}",
            "",
            "> **Tier 1** → SVG code patched directly (color/style/minor geometry)  ",
            "> **Tier 2** → primitive_graph.json rebuilt (structural/occlusion errors)",
        ]
    else:
        report_lines.append("_N/A — verdict is PASS_")

    report_lines += [
        "",
        "---",
        "## 7. Prompt sent to LLM (for prompt debugging)",
        "",
        "```",
        sys_msg_sent[:3000] + ("…(truncated)" if len(sys_msg_sent) > 3000 else ""),
        "```",
    ]

    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"📋 [Critic Debug] Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────
# 16. Hybrid Critic  (batch_vision_eval)
# ─────────────────────────────────────────────────────────────
def batch_vision_eval_tool(candidate_files: str, context: str,
                            reward_ablation: str = "full",
                            use_textgrad: bool = True,
                            generation_mode: str = "one_shot") -> str:
    """
    Hybrid Critic: three signal streams:
      1. CLIP + HPSv2 (scorer_service) — controlled by reward_ablation
      2. Deterministic preflight + graph constraint check
      3. LLM vision scoring
      4. TextGrad TGD (optional, controlled by use_textgrad)
    """
    manifest_files, _ = load_candidate_manifest()
    files = manifest_files or [f.strip() for f in candidate_files.split(",") if f.strip()]
    if not files:
        return "[Error] No candidate files to evaluate."

    user_task = file_io_tool("load", "user_task.txt")
    if "[Error]" in user_task: user_task = "Unknown task."

    graph_text = file_io_tool("load", PRIMITIVE_GRAPH_FILE)
    ref_tree   = file_io_tool("load", BEST_SVG_TREE_FILE)
    if "[Error]" in ref_tree: ref_tree = ""

    # ── Render all candidates to PNG ─────────────────────────
    valid_candidates = []
    for fname in files:
        abs_svg = get_abs_path(fname)
        abs_png = abs_svg.replace(".svg",".png")
        if not os.path.exists(abs_svg):
            continue
        try:
            svg_code = file_io_tool("load", fname)
            svg_to_png_with_browser(abs_svg, abs_png, 256, 256)
            with open(abs_png,"rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            valid_candidates.append({
                "file":     fname,
                "svg_code": svg_code,
                "png_path": abs_png,
                "b64":      b64,
                "tree":     extract_svg_tree_to_json(svg_code),
            })
        except Exception as e:
            print(f"[Critic] Failed to render {fname}: {e}")

    if not valid_candidates:
        return "[Error] Could not render any candidate SVG."

    # ── Signal 1: metric scores ───────────────────────────────
    use_clip = reward_ablation not in ("no_clip","no_clip_no_hps","llm_only")
    use_hps  = reward_ablation not in ("no_hps", "no_clip_no_hps","llm_only")
    for c in valid_candidates:
        if use_clip or use_hps:
            scores = compute_metric_scores(c["png_path"], user_task)
            c["clip"] = scores["clip"] if use_clip else None
            c["hps"]  = scores["hps"]  if use_hps  else None
        else:
            c["clip"] = None; c["hps"] = None

    # ── Select best candidate + compute metric_score ─────────────────────────
    # Both candidate selection and current_best ranking use the same function:
    # primary key  = CLIP score bucketed to 0.01 precision
    #                (CLIP measures text-image semantic alignment)
    # secondary key = HPS score (tiebreaker only, aesthetic preference)
    #
    # Using a tuple (clip_bucket, hps) instead of a weighted sum means HPS can
    # only break ties within the same CLIP bucket — it can never override a
    # meaningfully better CLIP score.  This matches your stated preference.
    #
    # llm_only ablation: both signals are None → fall back to LLM score / 10.
    def _candidate_metric(c) -> tuple:
        clip = c["clip"]
        hps  = c["hps"]
        if clip is not None:
            clip_bucket = round(clip * 100)          # e.g. 0.313 → 31
            hps_val     = float(hps) if hps is not None else 0.0
            return (clip_bucket, hps_val)
        elif hps is not None:
            # HPS-only ablation: treat as primary
            return (round(hps * 100), 0.0)
        else:
            # llm_only: no metric signals, all candidates tied at (0, 0)
            return (0, 0.0)

    best = max(valid_candidates, key=_candidate_metric) if len(valid_candidates) > 1 \
           else valid_candidates[0]

    # metric_score is a scalar derived from the same priority logic, used for
    # cross-run comparison in BEST_SCORE_FILE.  We convert back to a float so
    # comparisons like metric_score > stored_score work correctly.
    # Encoding: clip * 1.0 + hps * 0.001  — this preserves the CLIP-first
    # ordering as a single float without losing HPS as a fine-grained tiebreaker.
    def _to_metric_score(clip, hps, llm_score) -> float:
        if clip is not None:
            return float(clip) + (float(hps) if hps is not None else 0.0) * 0.001
        elif hps is not None:
            return float(hps)
        else:
            return llm_score / 10.0

    # ── Signal 2: deterministic preflight + graph constraints ─────────────
    preflight_blocking, preflight_warnings = svg_static_preflight(
        best["svg_code"], best["png_path"])
    graph_blocking, graph_warnings = validate_graph_constraints(
        best["svg_code"], graph_text)

    # Promote graph warnings that indicate structural failure to blocking.
    # inside_bounds_of / contains violations mean an element is visually
    # outside its declared container — this is a real rendering defect, not
    # a cosmetic issue. We tolerate up to 1 violation (bbox mismatch can
    # sometimes be a measurement artefact from path-bbox approximation),
    # but 2+ is a signal the SVG_Engineer ignored the graph topology.
    # Promote graph warnings to blocking only when they indicate real structural
    # failures — not measurement artefacts.
    #
    # Exclusion rule: if a node's bbox is None (e.g. polygon with missing/empty
    # points), the inside_bounds_of check is unreliable and must NOT be promoted.
    # We detect this by checking whether the violation string references a node
    # that already has a polygon_missing_points blocking issue.
    missing_points_nodes = set()
    for b in graph_blocking:
        if b.startswith("polygon_missing_points:"):
            # extract node id: "polygon_missing_points: <id> — ..."
            parts = b.split(":")
            if len(parts) >= 2:
                nid = parts[1].strip().split(" ")[0]
                missing_points_nodes.add(nid)

    GRAPH_WARNING_BLOCKING_THRESHOLD = 2
    structural_violations = []
    for w in graph_warnings:
        # Skip inside_bounds_of violations where the container is a polygon with
        # missing points — the bbox measurement is unreliable in that case.
        if w.startswith("inside_bounds_of_violation"):
            # format: "inside_bounds_of_violation: <source>-><target>"
            target = w.split("->")[-1].strip() if "->" in w else ""
            if target in missing_points_nodes:
                continue  # measurement artefact, skip
        if (w.startswith("inside_bounds_of_violation")
                or w.startswith("contains_violation")
                or w.startswith("occludes_doc_order_violation")):
            structural_violations.append(w)

    if len(structural_violations) >= GRAPH_WARNING_BLOCKING_THRESHOLD:
        for v in structural_violations:
            promoted = f"structural_violation_promoted: {v}"
            if promoted not in graph_blocking:
                graph_blocking.append(promoted)

    all_blocking = preflight_blocking + graph_blocking
    all_warnings = preflight_warnings + graph_warnings

    # ── Signal 3: LLM vision scoring ──────────────────────────────────────
    metric_context = ""
    if best["clip"] is not None:
        metric_context += f"CLIP score: {best['clip']:.3f}\n"
    if best["hps"] is not None:
        metric_context += f"HPSv2 score: {best['hps']:.3f}\n"

    # Include graph warnings in LLM prompt so it can factor structural
    # constraint violations into its visual assessment even when they don't
    # rise to blocking level.
    warnings_context = ""
    if all_warnings:
        warnings_context = (
            "\n=== GRAPH / PREFLIGHT WARNINGS (non-blocking) ===\n"
            + "\n".join(f"- {w}" for w in all_warnings)
            + "\nThese did not block evaluation but may indicate rendering issues "
            "you should look for in the image.\n"
        )

    sys_msg = (
        "You are a strict Art Director and SVG QA inspector.\n\n"
        f"=== TASK ===\n{user_task}\n\n"
        f"=== PRIMITIVE GRAPH ===\n{graph_text}\n\n"
        f"=== REFERENCE BEST SVG TREE ===\n{ref_tree or 'None yet'}\n\n"
        f"=== CURRENT SVG TREE ===\n{best['tree']}\n\n"
        f"=== METRIC SCORES ===\n{metric_context or 'Not available'}\n\n"
        f"=== DETERMINISTIC BLOCKING ISSUES ===\n{json.dumps(all_blocking)}\n"
        f"{warnings_context}\n"
        "Score 1-10 (overall_svg_score). "
        "MUST be ≤ 4 if any blocking issue present.\n"
        "If warnings mention inside_bounds_of or occlusion violations, "
        "check the image carefully for elements outside their containers "
        "or wrong layering, and reflect this in your scores.\n"
        "Return JSON:\n"
        '{"overall_svg_score":int,"scores":{"Structure":int,"Occlusion":int,"Color":int,'
        '"Fidelity":int,"Polish":int},"pass_ready":bool,'
        '"blocking_issues":[...],"aggregated_advice":"..."}'
    )
    base_critic_sys_msg = sys_msg
    sys_msg = _runtime_critic_sys_msg(base_critic_sys_msg)
    llm_result: Dict = {}
    llm_result_raw: str = ""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":sys_msg},
                {"role":"user","content":[
                    {"type":"text","text":f"Context: {context}"},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{best['b64']}"}}
                ]}
            ],
            temperature=0.1,
            response_format={"type":"json_object"},
        )
        llm_result_raw = resp.choices[0].message.content or ""
        llm_result = json.loads(llm_result_raw)
    except Exception as e:
        print(f"[Critic] LLM scoring failed: {e}")

    overall_score  = max(1, min(10, int(llm_result.get("overall_svg_score",0) or 0)))
    llm_blocking   = [str(x).strip() for x in llm_result.get("blocking_issues",[]) if str(x).strip()]
    pass_ready     = bool(llm_result.get("pass_ready",False))
    advice         = llm_result.get("aggregated_advice","")

    all_blocking.extend(x for x in llm_blocking if x not in all_blocking)
    if all_blocking:
        pass_ready = False

    # ── Metric floor checks ───────────────────────────────────────────────
    # CLIP and HPS floors act as hard gates independent of LLM scoring.
    # Only applied when the signals are actually enabled (not ablated away).
    # Thresholds are intentionally loose — they catch catastrophic failures
    # (blank render, completely wrong subject) not minor quality issues.
    CLIP_PASS_FLOOR = 0.20   # below this = semantically wrong subject
    HPS_PASS_FLOOR  = 0.18   # below this = visually broken / near-blank
    clip_pass = (not use_clip) or (best["clip"] is not None and best["clip"] >= CLIP_PASS_FLOOR)
    hps_pass = (not use_hps) or (best["hps"] is not None and best["hps"] >= HPS_PASS_FLOOR)
    metric_pass = clip_pass and hps_pass

    if use_clip and not clip_pass:
        all_blocking.append(
            f"clip_score_too_low: {(best['clip'] or 0.0):.3f} < {CLIP_PASS_FLOOR} "
            f"(semantic alignment failure)")
    if use_hps and not hps_pass:
        all_blocking.append(
            f"hps_score_too_low: {(best['hps'] or 0.0):.3f} < {HPS_PASS_FLOOR} "
            f"(visual quality failure)")

    # ── Pass gate ─────────────────────────────────────────────────────────
    # PASS requires LLM readiness, no blocking issues, and enabled metric floors.
    verdict = "PASS" if (pass_ready and not all_blocking and metric_pass) else "FAIL"

    # ── Metric score for this run ─────────────────────────────────────────
    metric_score = _to_metric_score(best["clip"], best["hps"], overall_score)
    rollback_result = maybe_rollback_prompt_overrides(metric_score)
    if rollback_result:
        print(f"[TextGrad] {rollback_result}")

    retry_counter = RetryCounter()

    # Compute drift against current best (empty string if no best exists yet)
    _current_best_raw = file_io_tool("load", "current_best_icon.svg")
    _current_best_svg = "" if "[Error]" in _current_best_raw else _current_best_raw
    drift_info  = compute_structural_drift(_current_best_svg, best["svg_code"])
    drift_score = drift_info.get("drift_score", 0)
    tier        = diagnose_failure_tier(
        best["clip"] or 0.0, graph_blocking, drift_score,
        preflight_blocking, llm_blocking)

    # tgd_result must be defined before the PASS/FAIL branch so result_payload
    # can always reference it.
    tgd_result = ""

    if verdict == "PASS":
        retry_counter.reset()
        finalize_candidate_artifacts(
            best["file"],
            svg_tree_json=best["tree"],
            score_payload={"score": metric_score, "verdict": verdict,
                           "llm_score": overall_score},
        )
        _update_history_queue(best["svg_code"], best["png_path"],
                               overall_score, advice)
    else:
        retry_counter.consume()

        if use_textgrad and _TEXTGRAD_AVAILABLE:
            tgd_result = textgrad_prompt_rewrite(
                user_task=user_task,
                critic_sys_msg=base_critic_sys_msg,
                clip_score=best["clip"] or 0.0,
                hps_score=best["hps"] or 0.0,
                tier=tier,
                blocking_issues=all_blocking,
                graph_warnings=graph_warnings,
                preflight_warnings=preflight_warnings,
                critic_advice=advice,
                structural_drift=drift_info,
            )
            print(f"[TextGrad] {tgd_result}")
            save_json_workspace_file(PROMPT_OVERRIDE_METRIC_FILE, {
                "metric_score": metric_score,
                "clip": best["clip"],
                "hps": best["hps"],
                "llm_score": overall_score,
            })

        # Always update current_best with this run's best candidate.
        # The caller (Critic) decides whether to retry — current_best just
        # tracks the most recent evaluated result regardless of score.
        finalize_candidate_artifacts(
            best["file"], best["tree"],
            {"score": metric_score, "verdict": verdict,
             "llm_score": overall_score})
        _update_history_queue(best["svg_code"], best["png_path"],
                               overall_score, advice)

        save_json_workspace_file(FAIL_COUNTER_FILE, {"count": retry_counter.count})

    tgd_graph_patched = False

    result_payload = {
        "verdict":           verdict,
        "overall_svg_score": overall_score,
        "pass_ready":        pass_ready,
        "metric_pass":       metric_pass,
        "clip_pass":         clip_pass,
        "hps_pass":          hps_pass,
        "clip_pass_floor":   CLIP_PASS_FLOOR if use_clip else None,
        "hps_pass_floor":    HPS_PASS_FLOOR if use_hps else None,
        "blocking_issues":   all_blocking,
        "graph_warnings":    graph_warnings,
        "preflight_warnings":preflight_warnings,
        "clip_score":        best["clip"],
        "hps_score":         best["hps"],
        "reward_ablation":   reward_ablation,
        "use_textgrad":      use_textgrad,
        "tier":              tier if verdict == "FAIL" else None,
        "tgd_result":        tgd_result if verdict == "FAIL" else "",
        "tgd_graph_patched": tgd_graph_patched if verdict == "FAIL" else False,
        "retry_counter":     retry_counter.status(),
        "advices":           advice,
        "selected_candidate":best["file"],
    }

    _save_critic_debug_report(
        result_payload   = result_payload,
        best             = best,
        all_candidates   = valid_candidates,
        preflight_blocking = preflight_blocking,
        preflight_warnings = preflight_warnings,
        graph_blocking   = graph_blocking,
        graph_warnings   = graph_warnings,
        llm_result_raw   = llm_result_raw,
        sys_msg_sent     = sys_msg,
        reward_ablation  = reward_ablation,
        use_textgrad     = use_textgrad,
    )

    return f"[Vision Result] {json.dumps(result_payload, ensure_ascii=False)}"

def _update_history_queue(svg_code: str, png_path: str,
                           score: int, suggestion: str):
    queue: List[Dict] = []
    if os.path.exists(HISTORY_QUEUE_FILE):
        try:
            with open(HISTORY_QUEUE_FILE,"r",encoding="utf-8") as f:
                queue = json.load(f)
        except Exception:
            pass
    queue.append({"score":score,"suggestion":suggestion,
                  "png_filepath":png_path,"svg_code":svg_code})
    queue = queue[-3:]
    with open(HISTORY_QUEUE_FILE,"w",encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
# 17. Memory
# ─────────────────────────────────────────────────────────────
def update_memory_on_success(task_label: str) -> str:
    tree_text = file_io_tool("load", SEMANTIC_TREE_FILE)
    svg_text  = file_io_tool("load", "current_best_icon.svg")
    if "[Error]" in tree_text or "[Error]" in svg_text:
        return "[Error] Cannot update memory: missing files."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":
                 "You are a Principal Design Architect. "
                 "Return JSON: {\"lessons\": [\"lesson1\", ...]}"},
                {"role":"user","content":
                 f"Task: {task_label}\nTree: {tree_text[:1000]}\nSVG: {svg_text[:2000]}"}
            ],
            temperature=0.1,
            response_format={"type":"json_object"},
        )
        lessons = json.loads(resp.choices[0].message.content).get("lessons",[])
        mem_path = MEMORY_FILE
        data = {"design_suggestions":[]}
        if os.path.exists(mem_path):
            with open(mem_path,"r") as f: data = json.load(f)
        data["design_suggestions"].extend(lessons)
        data["design_suggestions"] = data["design_suggestions"][-8:]
        with open(mem_path,"w") as f: json.dump(data, f, indent=2)
        return f"[Success] {len(lessons)} lessons saved."
    except Exception as e:
        return f"[Error] Memory update: {e}"

# ─────────────────────────────────────────────────────────────
# 18. TOOLS_SCHEMA  +  TOOLS_FUNCTIONS
# ─────────────────────────────────────────────────────────────
TOOLS_SCHEMA = [
    {"type":"function","function":{
        "name":"file_io",
        "description":"Read (load) or write (save) workspace files.",
        "parameters":{"type":"object","properties":{
            "action":  {"type":"string","enum":["save","load"]},
            "filepath":{"type":"string"},
            "content": {"type":"string"},
        },"required":["action","filepath"]}}},

    {"type":"function","function":{
        "name":"python_run",
        "description":"Execute Python code for validation.",
        "parameters":{"type":"object","properties":{
            "code":{"type":"string"}
        },"required":["code"]}}},

    {"type":"function","function":{
        "name":"send_message",
        "description":"Send a message to another agent. THIS ENDS YOUR TURN.",
        "parameters":{"type":"object","properties":{
            "target": {"type":"string"},
            "message":{"type":"string"},
        },"required":["target","message"]}}},

    {"type":"function","function":{
        "name":"build_primitive_graph",
        "description":(
            "Build primitive_graph.json from semantic_tree.json. "
            "Call this before svg_gen_batch. "
            "Optionally pass graph_json_override for Tier-2 rebuild."),
        "parameters":{"type":"object","properties":{
            "graph_json_override":{"type":"string",
                "description":"Revised graph JSON from TGD Tier-2 (optional)."},
        }}}},

    {"type":"function","function":{
        "name":"svg_gen_batch",
        "description":(
            "Generate k SVG candidates from primitive_graph.json. "
            "Requires build_primitive_graph to have run first."),
        "parameters":{"type":"object","properties":{
            "k":        {"type":"integer","description":"Number of candidates (1-3)."},
            "feedback": {"type":"string", "description":"Critic feedback for retry."},
            "mode":     {"type":"string","enum":["one_shot","autoregressive"],
                         "description":"Generation mode."},
        }}}},

    {"type":"function","function":{
        "name":"batch_vision_eval",
        "description":"Evaluate SVG candidates with hybrid Critic (CLIP/HPS + LLM + TGD).",
        "parameters":{"type":"object","properties":{
            "candidate_files":{"type":"string"},
            "context":        {"type":"string"},
        },"required":["candidate_files","context"]}}},

    {"type":"function","function":{
        "name":"finish_task",
        "description":"Conclude the project after Critic PASS.",
        "parameters":{"type":"object","properties":{
            "summary":            {"type":"string"},
            "best_candidate_file":{"type":"string"},
        },"required":["summary"]}}},
]

# Runtime flags injected at startup (MASEnvironment sets these)
_RUNTIME_GENERATION_MODE = "one_shot"
_RUNTIME_REWARD_ABLATION  = "full"
_RUNTIME_USE_TEXTGRAD     = True

TOOLS_FUNCTIONS: Dict = {
    "file_io":             file_io_tool,
    "python_run":          python_run_tool,
    "send_message":        send_message_tool,
    "build_primitive_graph": lambda graph_json_override="": build_primitive_graph_tool(
                                graph_json_override),
    "svg_gen_batch":       lambda k=3, feedback="", mode=None: svg_gen_batch_tool(
                                k, feedback,
                                mode or _RUNTIME_GENERATION_MODE),
    "batch_vision_eval":   lambda candidate_files="", context="": batch_vision_eval_tool(
                                candidate_files, context,
                                _RUNTIME_REWARD_ABLATION,
                                _RUNTIME_USE_TEXTGRAD,
                                _RUNTIME_GENERATION_MODE),
}

# ─────────────────────────────────────────────────────────────
# 19. AgentNode
# ─────────────────────────────────────────────────────────────
class AgentNode:
    def __init__(self, name: str, role_desc: str,
                 allowed_tools: List[str], allowed_partners: List[str]):
        self.name            = name
        self.role_desc       = role_desc
        self.base_role_desc  = role_desc
        PROMPT_BASELINES[name] = role_desc
        self.allowed_tools   = allowed_tools
        self.allowed_partners= allowed_partners

    # ── Internal helpers ──────────────────────────────────────
    def _call_api(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=0.1,
            )
            msg = resp.choices[0].message
            return {"content": msg.content or "", "tool_calls": msg.tool_calls or []}
        except APIError as e:
            print(f"[{self.name}] API error: {e}")
            return {"content":"","tool_calls":[]}

    def _read_inbox(self) -> str:
        p = get_abs_path(f"inbox/{self.name}.md")
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f: content = f.read()
            os.remove(p)
            return content
        return "No new messages."

    def _build_workspace_context(self) -> str:
        files = WORKSPACE_CONTEXT_SHARED_FILES + WORKSPACE_CONTEXT_AGENT_FILES.get(self.name,[])
        chunks = []
        seen: set = set()
        for fname in files:
            if fname in seen: continue
            seen.add(fname)
            p = get_abs_path(fname)
            if not os.path.isfile(p): continue
            try:
                with open(p,"r",encoding="utf-8") as f: content = f.read()
            except Exception: continue
            if content.strip():
                chunks.append(f"\n### {fname} ###\n```\n{content}\n```\n")
        return "".join(chunks) or "(no workspace context)"

    def _save_debug_log(self, turn: int, step: int, messages: List, resp: Dict):
        os.makedirs(".debug_logs", exist_ok=True)
        p = os.path.join(".debug_logs", f"Turn_{turn:02d}_{self.name}.md")
        with open(p,"a",encoding="utf-8") as f:
            f.write(f"## Step {step+1}\n### Input\n```json\n"
                    f"{json.dumps(messages,indent=2,ensure_ascii=False)}\n```\n"
                    f"### Response\n```json\n"
                    f"{json.dumps(resp,indent=2,default=str)}\n```\n---\n\n")

    def _tool_call_parts(self, tc) -> Optional[Tuple[str, str, str]]:
        """Return (id, name, arguments) for SDK objects or dict tool calls."""
        if not tc or isinstance(tc, bool):
            return None
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                return None
            name = fn.get("name")
            args = fn.get("arguments", "{}")
            tid = tc.get("id", "")
        else:
            fn = getattr(tc, "function", None)
            if not fn:
                return None
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", "{}")
            tid = getattr(tc, "id", "")
        if not name:
            return None
        return str(tid), str(name), args if isinstance(args, str) else json.dumps(args)

    # ── Main step ─────────────────────────────────────────────
    def step(self, global_bus: GlobalEventManager, current_turn: int,
             debug_mode: bool = False) -> Tuple[Optional[str], bool]:
        print(f"\n[{self.name}] Turn {current_turn} — reading inbox...")
        inbox   = self._read_inbox()
        context = (f"=== GLOBAL EVENTS ===\n{global_bus.get_summary()}\n\n"
                   f"=== INBOX ===\n{inbox}\n\n")
        agent_tools = [t for t in TOOLS_SCHEMA
                       if t["function"]["name"] in self.allowed_tools]
        runtime_role_desc = _runtime_role_desc(self.name, self.base_role_desc)
        system_prompt = (
            f"You are {self.name}. {runtime_role_desc}\n\n"
            f"Turn: {current_turn}. Allowed partners: {self.allowed_partners}.\n"
            "RULES: 1) Read inbox. 2) Call ONE tool per step. "
            "3) End turn with send_message."
        )
        ws = self._build_workspace_context()
        messages = [
            {"role":"system","content": system_prompt},
            {"role":"system","content": f"=== WORKSPACE ===\n{ws}"},
            {"role":"user",  "content": f"{context}\nDecide next action."},
        ]
        svg_gen_done = False

        for local_step in range(10):  # max local tool calls per turn
            resp       = self._call_api(messages, agent_tools)
            tool_calls = resp.get("tool_calls", [])
            content    = resp.get("content","")

            if len(tool_calls) > 1:
                tool_calls = [tool_calls[0]]  # enforce one-at-a-time

            if debug_mode:
                self._save_debug_log(current_turn, local_step, messages, resp)

            asst_msg: Dict = {"role":"assistant","content": content or ""}
            if tool_calls:
                normalized_calls = []
                for tc in tool_calls:
                    parts = self._tool_call_parts(tc)
                    if not parts:
                        continue
                    tid, name, args_text = parts
                    normalized_calls.append({
                        "id": tid,
                        "type": "function",
                        "function": {"name": name, "arguments": args_text},
                    })
                asst_msg["tool_calls"] = normalized_calls
                tool_calls = normalized_calls
            messages.append(asst_msg)

            if not tool_calls:
                messages.append({"role":"user","content":
                    "WARNING: You MUST call a tool or send_message to end your turn."})
                continue

            yield_to   = self.name
            is_done    = False
            turn_ended = False

            for tc in tool_calls:
                parts = self._tool_call_parts(tc)
                if not parts:
                    continue
                tc_id, tname, arg_text = parts
                try:
                    args = json.loads(arg_text)
                except Exception as e:
                    messages.append({"role":"tool","tool_call_id":tc_id,
                                     "name":tname,"content":f"[Error] JSON parse: {e}"})
                    continue

                # ── send_message ──────────────────────────────
                if tname == "send_message" and tname in self.allowed_tools:
                    target  = args.get("target","")
                    body    = args.get("message","")
                    if target in self.allowed_partners:
                        send_message_tool(self.name, target, body)
                        global_bus.log_event(f"[{self.name}]→[{target}]: handoff")
                        result_str = f"[Success] Message to {target}."
                        yield_to   = target
                        turn_ended = True
                    else:
                        result_str = f"[Error] '{target}' not in allowed partners."

                # ── finish_task ───────────────────────────────
                elif tname == "finish_task" and tname in self.allowed_tools:
                    best_file    = args.get("best_candidate_file","")
                    summary_hint = args.get("summary","")
                    # Finalize if artifacts not yet present
                    if not os.path.exists(get_abs_path("current_best_icon.svg")):
                        resolved = resolve_final_candidate(best_file)
                        if resolved:
                            finalize_candidate_artifacts(resolved)
                    mem = update_memory_on_success(
                        file_io_tool("load","user_task.txt"))
                    print(f"\n🧠 [MEMORY]: {mem}")
                    print(f"\n🎉 [DONE by {self.name}]: {summary_hint}")
                    messages.append({"role":"tool","tool_call_id":tc_id,
                                     "name":tname,"content":f"[Done] {summary_hint}"})
                    global_bus.log_event(f"[{self.name}] finished task.")
                    return None, True

                # ── regular tools ─────────────────────────────
                elif tname in TOOLS_FUNCTIONS and tname in self.allowed_tools:
                    # File-write permission check
                    if tname == "file_io" and args.get("action") == "save":
                        ok, err = validate_agent_file_write(
                            self.name, args.get("filepath",""))
                        if not ok:
                            messages.append({"role":"tool","tool_call_id":tc_id,
                                             "name":tname,"content": err})
                            continue
                    # Prevent duplicate svg_gen_batch in same turn
                    if tname == "svg_gen_batch":
                        if svg_gen_done:
                            messages.append({"role":"tool","tool_call_id":tc_id,
                                "name":tname,"content":
                                "[Error] svg_gen_batch already ran this turn. "
                                "Send results to Critic instead."})
                            continue
                        svg_gen_done = True
                    try:
                        result = TOOLS_FUNCTIONS[tname](**args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"[Error] {tname}: {e}"
                    global_bus.log_event(f"[{self.name}] used {tname}")
                else:
                    result_str = f"[Error] Tool '{tname}' not allowed for {self.name}."

                messages.append({"role":"tool","tool_call_id":tc_id,
                                 "name":tname,"content": result_str})

            if turn_ended:
                return yield_to, is_done

        # Exceeded local steps — emergency handoff to Coordinator
        print(f"[{self.name}] Max local steps reached without send_message.")
        send_message_tool("System","Coordinator",
                          f"{self.name} exhausted steps. Re-routing.")
        return "Coordinator", False

# ─────────────────────────────────────────────────────────────
# 20. MASEnvironment  (3-agent orchestrator)
# ─────────────────────────────────────────────────────────────
class MASEnvironment:
    def __init__(self, agents: List[AgentNode], debug_mode: bool = False):
        self.agents     = {a.name: a for a in agents}
        self.global_bus = GlobalEventManager()
        self.debug_mode = debug_mode

        # Reset workspace (keep memory if it exists)
        mem_backup = None
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE,"r",encoding="utf-8") as f:
                mem_backup = f.read()
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)
        os.makedirs(WORKSPACE_DIR)
        os.makedirs(get_abs_path("inbox"))
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE,"w",encoding="utf-8") as f:
                json.dump({"design_suggestions":[]}, f)
        if mem_backup:
            with open(MEMORY_FILE,"w",encoding="utf-8") as f:
                f.write(mem_backup)

        if debug_mode:
            if os.path.exists(".debug_logs"):
                shutil.rmtree(".debug_logs")
            os.makedirs(".debug_logs")

    def run(self, user_task: str, start_agent: str, max_turns: int = 30):
        print(f"\n{'='*55}")
        print(f"[INIT] Pipeline. Debug={self.debug_mode}")
        print(f"Task: {user_task[:80]}...")
        print(f"{'='*55}\n")

        file_io_tool("save", "user_task.txt", user_task)
        send_message_tool("User", start_agent,
                          f"New task. Read user_task.txt and generate semantic_tree.json first.")
        current = start_agent

        for turn in range(1, max_turns+1):
            print(f"\n{'─'*50}")
            print(f"Turn {turn}/{max_turns} | Agent: \033[92m{current}\033[0m")
            print(f"{'─'*50}")

            agent = self.agents.get(current)
            if not agent:
                print(f"[Fatal] Agent '{current}' not found → Coordinator")
                current = "Coordinator"
                continue

            next_agent, is_done = agent.step(self.global_bus, turn, self.debug_mode)

            if is_done:
                print("\n\033[93m=== PIPELINE COMPLETE ===\033[0m")
                break

            # Progress watchdog: intervene at 50% and 75%
            if turn in [max_turns//2, (3*max_turns)//4]:
                ratio = turn/max_turns*100
                print(f"\n\033[91m=== WARNING {ratio:.0f}% turns used ===\033[0m")
                current = "Coordinator"
                send_message_tool("System","Coordinator",
                                  f"Pipeline at {ratio:.0f}% of max turns ({turn}/{max_turns}). "
                                  "Check for loops and intervene.")
            elif next_agent in self.agents:
                current = next_agent
            else:
                print(f"[Warning] Invalid next agent '{next_agent}' → Coordinator")
                current = "Coordinator"

        if turn >= max_turns:
            print(f"\n\033[91m[STOP] Max turns reached.\033[0m")

# ─────────────────────────────────────────────────────────────
# 21. Agent definitions
# ─────────────────────────────────────────────────────────────
def classify_failure_mode(task_text: str) -> str:
    """
    Classify why a task likely failed based on best_score.json and
    the last critic debug report.  Used for ablation experiment analysis.

    Categories:
      CAPABILITY_LIMIT   — task requires geometry the LLM cannot render
                           (wrap-around paths, precise organic connections,
                           deep occlusion stacks)
      LOW_METRIC         — CLIP or HPS stayed below floor across all retries
      STRUCTURAL_DRIFT   — SVG_Engineer kept deviating from the graph
      MAX_RETRIES        — reached retry limit without resolving blocking issues
      UNKNOWN            — could not determine cause
    """
    score_data   = load_json_file(BEST_SCORE_FILE, {})
    best_clip    = score_data.get("score", 0.0)   # metric_score = clip + hps*0.001

    # Heuristic: metric_score encodes clip as the integer part
    # If clip < 0.22 across all retries → semantically wrong → capability limit
    # or subject understanding failure
    if best_clip < 0.22:
        return "LOW_METRIC_CLIP"

    # Check last critic report for clues
    last_report = ""
    try:
        run_dirs = sorted(Path(CRITIC_DEBUG_DIR).iterdir(), key=lambda p: p.name)
        if run_dirs:
            report_path = run_dirs[-1] / "report.md"
            if report_path.exists():
                last_report = report_path.read_text(encoding="utf-8")
    except Exception:
        pass

    if "wrap" in last_report.lower() or "cradle" in last_report.lower() \
            or "c-shaped" in last_report.lower():
        return "CAPABILITY_LIMIT_WRAP_PATH"

    if "structural_drift" in last_report.lower():
        return "STRUCTURAL_DRIFT"

    if "3/3" in last_report or "max retries" in last_report.lower():
        return "MAX_RETRIES"

    # Fall back to keyword check on task text itself
    hard_words = ["wrap", "cradle", "curl", "coil", "encircle", "surround",
                  "spiral", "weave", "intertwine"]
    if any(w in task_text.lower() for w in hard_words):
        return "CAPABILITY_LIMIT_COMPLEX_GEOMETRY"

    return "UNKNOWN"

def build_agents(generation_mode: str, reward_ablation: str,
                 use_textgrad: bool) -> List[AgentNode]:
    """Inject runtime flags and return the 3+1 agent list."""
    global _RUNTIME_GENERATION_MODE, _RUNTIME_REWARD_ABLATION, _RUNTIME_USE_TEXTGRAD
    _RUNTIME_GENERATION_MODE = generation_mode
    _RUNTIME_REWARD_ABLATION  = reward_ablation
    _RUNTIME_USE_TEXTGRAD     = use_textgrad

    coordinator = AgentNode(
        name="Coordinator",
        role_desc=(
            "You are the thin watchdog coordinator. You ONLY intervene when:\n"
            "  a) Another agent reports an unrecoverable error.\n"
            "  b) The system watchdog routes you because turn budget is running low.\n"
            "  c) An agent is stuck in a loop.\n"
            "In normal operation the pipeline is: TreeBuilder → SVG_Engineer → Critic.\n"
            "Do NOT interrupt the pipeline unless one of the above conditions applies.\n"
            "If the pipeline appears stuck, command the appropriate agent to restart its step.\n"
            "If Critic reports PASS, command the relevant agent to call finish_task."
        ),
        allowed_tools=["file_io","send_message","finish_task"],
        allowed_partners=["TreeBuilder","SVG_Engineer","Critic"],
    )

    tree_builder = AgentNode(
        name="TreeBuilder",
        role_desc=(
            "You are the Semantic Architect. Your sole output is 'semantic_tree.json'.\n\n"
            "SCHEMA (version '2.0'):\n"
            "{\n"
            '  "version": "2.0",\n'
            '  "canvas": {"viewBox": "0 0 256 256"},\n'
            '  "material_library": {"mat_id": {"fill":"#hex","stroke":"#hex",...}},\n'
            '  "tree": { ...recursive tree... }\n'
            "}\n\n"
            "TREE NODE RULES:\n"
            "- Every node must have: id (unique snake_case), role (semantic string).\n"
            "- ONLY leaf nodes (is_leaf: true) get a 'geometry' block with:\n"
            '    bbox_norm: [x, y, w, h] in 0..1 coordinates,\n'
            '    shape_type: "rect"|"circle"|"ellipse"|"path"|"polygon",\n'
            "    material_ref: matching a material_library key,\n"
            "    importance: 'primary'|'secondary'.\n"
            "- Non-leaf nodes are ONLY semantic grouping containers — no geometry.\n"
            "- Declare ALL logically necessary parts. Symmetric objects need ALL instances.\n\n"
            "MATERIAL LIBRARY RULES:\n"
            "- 'fill' must be a plain hex color (#rrggbb or #rgb) or 'none'. "
            "NEVER write CSS gradient syntax like 'radial-gradient(...)' or "
            "'linear-gradient(...)' in a fill field — SVG does not support this.\n"
            "- For gradient fills, add a 'gradient_ref' field instead with a "
            "short descriptive id (e.g. 'grad_glow'). The SVG_Engineer will "
            "create the matching <radialGradient> or <linearGradient> in <defs>.\n"
            "  Example: {\"fill\": \"url(#grad_glow)\", \"gradient_ref\": \"grad_glow\", "
            "\"gradient_type\": \"radial\", \"gradient_colors\": [\"#F1C40F\", \"transparent\"]}\n"
            "- If two adjacent nodes share the same fill color, add a 'notes' field "
            "instructing the engineer to add stroke/shadow to prevent visual merging.\n\n"
            "CAPABILITY CONSTRAINTS — these specific patterns cannot be rendered\n"
            "reliably and must be avoided:\n"
            "1. NO spatial wrapping: do not design a shape whose role is to\n"
            "   physically surround or cradle another shape from outside\n"
            "   (e.g. a C-shaped body enclosing a circle, a hand gripping a ball).\n"
            "   Use overlapping layered shapes instead — the illusion of wrapping\n"
            "   comes from correct draw_order, not from a single path outline.\n"
            "2. NO pixel-precise path connections: do not add notes requiring two\n"
            "   separate nodes to meet at an exact coordinate\n"
            "   (e.g. 'tail_main must connect to torso at exactly [0.3, 0.7]').\n"
            "   Overlapping bboxes with the same fill color fuse seamlessly.\n"
            "3. NO more than 3 levels of occlusion in the same spatial region:\n"
            "   background → mid → foreground is the safe maximum.\n"
            "4. For 'path' shape_type: it is fine for curved silhouettes, tails,\n"
            "   and organic outlines. Avoid using path when the geometry requires\n"
            "   it to spatially wrap or contain another node (see constraint 1).\n\n"
            "CRITICAL:\n"
            "- Ensure bbox_norm values are valid: x+w <= 1, y+h <= 1, all >= 0.\n"
            "- Save ONLY valid JSON to semantic_tree.json — no markdown fences.\n\n"
            "NORMAL START: Read user_task.txt, generate semantic_tree.json, then\n"
            "  send_message to SVG_Engineer: 'semantic_tree.json is ready.'\n\n"
            "TIER-2 REBUILD (when Critic sends you feedback):\n"
            "  Critic routes Tier-2 failures to you because the graph structure itself\n"
            "  is wrong — missing nodes, wrong occlusion topology, fundamentally\n"
            "  misplaced elements (e.g. 'cat is missing a leg', 'head and body reversed').\n"
            "  When this happens:\n"
            "  1. Read the Critic feedback carefully.\n"
            "  2. Load the existing semantic_tree.json and revise only the specific\n"
            "     nodes/structure described in the feedback. Do NOT redesign from\n"
            "     scratch unless feedback says the entire concept is wrong.\n"
            "  3. Save the revised semantic_tree.json.\n"
            "  4. send_message to SVG_Engineer: 'semantic_tree.json revised per Critic\n"
            "     Tier-2 feedback. Please rebuild graph and regenerate SVG.'"
        ),
        allowed_tools=["file_io","send_message"],
        allowed_partners=["SVG_Engineer"],
    )

    svg_engineer = AgentNode(
        name="SVG_Engineer",
        role_desc=(
            "You are the technical execution engine. You produce SVG from the semantic tree.\n\n"
            "WORKFLOW (follow strictly):\n"
            f"STEP 1: Call build_primitive_graph() — reads '{SEMANTIC_TREE_FILE}', builds\n"
            f"        '{PRIMITIVE_GRAPH_FILE}' with topology edges and draw_order.\n"
            "STEP 2: Call svg_gen_batch(k=3) — generates candidates from the primitive graph.\n"
            f"STEP 3: Load '{CANDIDATE_MANIFEST_FILE}' to confirm candidates are ready.\n"
            "STEP 4: Send candidates to Critic with token '[FINAL_SVG_REVIEW]'.\n\n"
            "TIER-1 RETRY (when Critic sends Tier-1 feedback directly to you):\n"
            "  Tier-1 = SVG code-level defects: wrong coordinates, clipPath issues,\n"
            "  visual gaps, color/stroke problems, minor misalignment.\n"
            "  Action: call svg_gen_batch(feedback=<critic_advice>, k=3) directly.\n"
            "  Do NOT touch the graph — the graph structure is correct.\n"
            "  TextGrad may have rewritten your runtime role prompt; follow that\n"
            "  strategy while preserving the graph and semantic tree.\n\n"
            "TEXTGRAD PROMPT REWRITE:\n"
            "  TGD may adjust runtime instructions only. It never patches SVG,\n"
            "  primitive_graph.json, or semantic_tree.json directly.\n\n"
            "TIER-2 REBUILD (when TreeBuilder sends you a revised tree):\n"
            "  TreeBuilder has already updated semantic_tree.json.\n"
            "  Action: call build_primitive_graph() first (rebuild graph from new tree),\n"
            "  then call svg_gen_batch(k=3) fresh — no feedback parameter needed.\n\n"
            "ERROR HANDLING:\n"
            "- If build_primitive_graph returns [Error]: report to Coordinator immediately.\n"
            "- If svg_gen_batch returns [Error]: report full error to Coordinator.\n"
            "- Do NOT loop on errors. Report once and wait."
        ),
        allowed_tools=["file_io","build_primitive_graph","svg_gen_batch",
                       "python_run","send_message"],
        allowed_partners=["Critic","Coordinator"],
    )

    critic = AgentNode(
        name="Critic",
        role_desc=(
            "You are the strict Art Director and QA gate.\n\n"
            "WORKFLOW:\n"
            "1. When SVG_Engineer sends '[FINAL_SVG_REVIEW]', call batch_vision_eval with\n"
            f"   candidate_files from '{CANDIDATE_MANIFEST_FILE}' and context='[FINAL_SVG_REVIEW]'.\n"
            "2. Inspect the returned JSON and route based on tier:\n\n"
            "   verdict='PASS':\n"
            "     → send_message to Coordinator: 'SVG PASS. Ready to finish.'\n\n"
            "   verdict='FAIL' + tier=1 (SVG code defects):\n"
            "     → send_message to SVG_Engineer with:\n"
            "       [REJECTED] Tier: 1 (SVG patch)\n"
            "       Blocking issues: <list>\n"
            "       Feedback: <aggregated_advice>\n"
            "       TGD Result: <tgd_result if any>\n"
            "       Retry Counter: <retry_counter>\n"
            "     Tier-1 issues include: wrong coords, clipPath misuse, visual gaps,\n"
            "     color/stroke problems, minor misalignment — anything fixable in SVG code\n"
            "     without changing the semantic structure.\n\n"
            "   verdict='FAIL' + tier=2 (structural/semantic defects):\n"
            "     TextGrad may rewrite runtime prompts, but it never patches tree/graph/SVG.\n"
            "     Always send_message to TreeBuilder with:\n"
            "          [TIER-2 REBUILD REQUIRED]\n"
            "          Blocking issues: <list>\n"
            "          Structural feedback: <aggregated_advice>\n"
            "          Retry Counter: <retry_counter>\n"
            "     Tier-2 issues: missing nodes, wrong occlusion, fundamentally wrong geometry.\n"
            "3. If retry_counter shows 3/3 (exhausted):\n"
            "   → send_message to Coordinator: 'Max retries reached. Finalizing best SVG.'\n\n"
            "IMPORTANT: Always pass the exact token '[FINAL_SVG_REVIEW]' in the context field.\n"
            "ROUTING RULE: Tier-1 → SVG_Engineer. "
            "Tier-2 → TreeBuilder. TextGrad only rewrites runtime prompts."
        ),
        allowed_tools=["file_io","batch_vision_eval","send_message"],
        allowed_partners=["Coordinator","SVG_Engineer","TreeBuilder"],
    )

    return [coordinator, tree_builder, svg_engineer, critic]

# ─────────────────────────────────────────────────────────────
# 22. Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text→Tree→Graph→SVG pipeline with ablation controls")
    parser.add_argument("--task",   type=str, default=None,
        help="Single task description (omit to use --jsonl)")
    parser.add_argument("--jsonl",  type=str,
        default="mmsvg_text_sampled.jsonl",
        help="Batch task JSONL path")
    parser.add_argument("--results_dir", type=str, default="results",
        help="Output directory for results")
    parser.add_argument("--max_turns", type=int, default=30,
        help="Max turns per task")
    parser.add_argument("--debug", action="store_true",
        help="Save per-step debug logs")
    parser.add_argument("--clean", action="store_true",
        help="Remove runtime/cache folders before running; preserves results/ and models/.")

    # ── Ablation flags ────────────────────────────────────────
    parser.add_argument("--generation_mode",
        choices=["one_shot","autoregressive"], default="one_shot",
        help=(
            "SVG generation strategy:\n"
            "  one_shot       = full SVG in a single LLM call (default)\n"
            "  autoregressive = generate one node at a time along draw_order"))
    parser.add_argument("--reward_ablation",
        choices=["full","no_clip","no_hps","no_clip_no_hps","llm_only"],
        default="full",
        help=(
            "Control which reward signals are active in Critic:\n"
            "  full            = CLIP + HPS + LLM + preflight (default)\n"
            "  no_clip         = HPS + LLM + preflight\n"
            "  no_hps          = CLIP + LLM + preflight\n"
            "  no_clip_no_hps  = LLM + preflight only\n"
            "  llm_only        = LLM scoring only (closest to original system)"))
    parser.add_argument("--no_textgrad", dest="use_textgrad",
        action="store_false", default=True,
        help="Disable TextGrad TGD; fall back to plain LLM feedback retry (ablation)")

    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        parser.error("OPENROUTER_API_KEY is required. Export it before running StructSVG.")

    if args.clean:
        clean_runtime_dirs()

    # ── TextGrad setup ────────────────────────────────────────
    if args.use_textgrad:
        ok = _setup_textgrad_engines()
        if not ok:
            print("[Warning] TextGrad setup failed; proceeding without TGD.")
            args.use_textgrad = False

    # ── Task list ─────────────────────────────────────────────
    if args.task:
        tasks = [{"text": args.task, "idx": 0}]
    else:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"[Fatal] JSONL file not found: {jsonl_path}")
            sys.exit(1)
        tasks = []
        with open(jsonl_path,"r",encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    tasks.append({"text": obj["text"], "idx": idx})
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Warning] Skip line {idx}: {e}")

    if not tasks:
        print("[Fatal] No tasks found."); sys.exit(1)

    results_dir   = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_file = results_dir / "progress.json"
    completed: set = set()
    if progress_file.exists():
        try:
            with open(progress_file,"r") as f:
                completed = set(json.load(f).get("completed",[]))
            print(f"[Resume] {len(completed)} tasks already done.")
        except Exception:
            pass

    print(f"\n🚀 [BATCH] {len(tasks)} tasks → {results_dir}")
    print(f"   generation_mode  : {args.generation_mode}")
    print(f"   reward_ablation  : {args.reward_ablation}")
    print(f"   use_textgrad     : {args.use_textgrad}\n")

    for task_obj in tasks:
        idx  = task_obj["idx"]
        text = task_obj["text"]

        if idx in completed:
            print(f"[Skip] Task {idx:03d}")
            continue

        print(f"\n{'='*60}")
        print(f"[Task {idx:03d}] {text[:80]}")
        print(f"{'='*60}")

        # Backup cross-task memory
        mem_backup = None
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE,"r",encoding="utf-8") as f:
                mem_backup = f.read()

        agents = build_agents(args.generation_mode,
                               args.reward_ablation,
                               args.use_textgrad)
        env = MASEnvironment(agents, debug_mode=args.debug)

        if mem_backup:
            with open(MEMORY_FILE,"w",encoding="utf-8") as f:
                f.write(mem_backup)

        reset_prompt_overrides_for_task()
        os.makedirs(PUBLISHED_DIR, exist_ok=True)

        try:
            env.run(user_task=text, start_agent="TreeBuilder",
                    max_turns=args.max_turns)
        except Exception as e:
            print(f"\n[Error] Task {idx:03d}: {e}")
            (results_dir/f"task_{idx:03d}_FAILED.txt").write_text(
                f"Task: {text}\nError: {e}\n", encoding="utf-8")
            continue

        # Copy final SVG/PNG to results
        final_svg = Path(PUBLISHED_DIR)/"current_best_icon.svg"
        safe = re.sub(r'[^\w\s-]','',text[:40]).strip().replace(' ','_')

        if final_svg.exists():
            dest = results_dir/f"task_{idx:03d}_{safe}.svg"
            shutil.copy2(final_svg, dest)
            print(f"✅ [Saved] {dest.name}")
            final_png = Path(PUBLISHED_DIR)/"current_best_icon.png"
            if final_png.exists():
                shutil.copy2(final_png,
                             results_dir/f"task_{idx:03d}_{safe}.png")
        else:
            print(f"⚠️  [Warning] Task {idx:03d} produced no final SVG.")

        # Write per-task metadata for ablation analysis
        score_data   = load_json_file(BEST_SCORE_FILE, {})
        fail_data    = load_json_file(FAIL_COUNTER_FILE, {})
        final_verdict = score_data.get("verdict", "UNKNOWN")
        task_meta = {
            "idx":              idx,
            "task":             text,
            "verdict":          final_verdict,
            "metric_score":     score_data.get("score", 0.0),
            "llm_score":        score_data.get("llm_score", 0),
            "retries_used":     fail_data.get("count", 0),
            "generation_mode":  args.generation_mode,
            "reward_ablation":  args.reward_ablation,
            "use_textgrad":     args.use_textgrad,
            "failure_mode":     (classify_failure_mode(text)
                                 if final_verdict != "PASS" else "NONE"),
            "svg_produced":     final_svg.exists(),
        }
        meta_path = results_dir / f"task_{idx:03d}_{safe}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(task_meta, f, ensure_ascii=False, indent=2)
        print(f"📊 [Meta] {meta_path.name}  verdict={final_verdict}  "
              f"metric={task_meta['metric_score']:.4f}  "
              f"failure_mode={task_meta['failure_mode']}")

        completed.add(idx)
        with open(progress_file,"w",encoding="utf-8") as f:
            json.dump({"completed": sorted(completed)}, f)

        if args.debug:
            try:
                from combine_logs import combine_markdown_files
                combine_markdown_files(".debug_logs")
            except Exception:
                pass

    print(f"\n\033[92m✅ [BATCH DONE] Results in: {results_dir}\033[0m")
