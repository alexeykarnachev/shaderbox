"""Build the THROWAWAY project for the 090 BEFORE baseline: three documents (trivial /
medium ~8ms GPU / heavy ~100ms GPU), authored directly on disk per dev_flow.md's
"Authoring / debugging documents directly" recipe — no App, no copilot.

Iteration counts were calibrated by `00_calibrate_shaders.py` on this box (RTX 3090,
1024x1024 f2 target): n=1300 -> ~8.0 ms, n=17000 -> ~99.1 ms (median of 15 finish()-bounded
draws). The heavy document reuses that exact fragment shader at document canvas size 1024x1024
so the calibration transfers.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/build_project.py <project_dir>`
Writes documents/{trivial,medium,heavy}/{document.json,passes/main.frag.glsl}, passes files
BEFORE document.json in each dir per the atomic-write rule.
"""

import json
import sys
from pathlib import Path

_MEDIUM_N = 1300
_HEAVY_N = 17000
_CANVAS = 1024

_TRIVIAL_FRAG = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() {
    fs_color = vec4(vs_uv, 0.5, 1.0);
}
"""

_NOISE_FRAG_TEMPLATE = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;

float hash(vec2 p) {{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}}

float noise(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}

void main() {{
    vec2 uv = vs_uv * 8.0;
    float acc = 0.0;
    float amp = 0.5;
    for (int i = 0; i < {n}; i++) {{
        acc += noise(uv) * amp;
        uv = uv * 2.03 + vec2(37.1, 91.7);
        amp *= 0.998;
    }}
    fs_color = vec4(vec3(acc), 1.0);
}}
"""


def _document_json(name: str, description: str) -> dict:
    return {
        "canvas_size": [_CANVAS, _CANVAS],
        "uniforms": {},
        "ui_state": {
            "ui_name": name,
            "description": description,
            "render_media_details": {
                "is_video": True,
                "file_details": {"path": "", "size": 0},
                "resolution_details": {"width": 0, "height": 0},
                "duration": 6.0,
                "fps": 30,
                "quality": 0,
            },
            "ui_uniforms": {},
        },
    }


def _write_document(root: Path, doc_id: str, frag: str, name: str, description: str) -> None:
    doc_dir = root / "documents" / doc_id
    passes_dir = doc_dir / "passes"
    passes_dir.mkdir(parents=True)
    # Passes BEFORE document.json (dev_flow.md atomic-write rule).
    (passes_dir / "main.frag.glsl").write_text(frag, encoding="utf-8")
    (doc_dir / "document.json").write_text(
        json.dumps(_document_json(name, description), indent=2), encoding="utf-8"
    )


def build(root: Path) -> None:
    (root / "documents").mkdir(parents=True, exist_ok=True)
    _write_document(
        root,
        "trivial",
        _TRIVIAL_FRAG,
        "Trivial",
        "090 baseline: a plain UV gradient, no loops.",
    )
    _write_document(
        root,
        "medium",
        _NOISE_FRAG_TEMPLATE.format(n=_MEDIUM_N),
        "Medium",
        f"090 baseline: {_MEDIUM_N} noise iterations, calibrated to ~8ms GPU at 1024x1024.",
    )
    _write_document(
        root,
        "heavy",
        _NOISE_FRAG_TEMPLATE.format(n=_HEAVY_N),
        "Heavy",
        f"090 baseline: {_HEAVY_N} noise iterations, calibrated to ~100ms GPU at 1024x1024.",
    )


if __name__ == "__main__":
    target = Path(sys.argv[1])
    build(target)
    print(f"built project at {target}")
