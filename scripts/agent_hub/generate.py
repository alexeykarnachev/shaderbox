"""Generate the agent-hub page: the full, holistic report on the copilot.

The maintainer's communication hub: one static page carrying the ENTIRE agent surface (system
prompt verbatim, conventions, every tool signature, every config knob with its doc comment, the
context-assembly model) plus every dogfood run's dialogue, media and verdicts. Regenerate with
`uv run python scripts/agent_hub/generate.py`; output is `scripts/agent_hub/site/` (gitignored —
static HTML + copied media, servable by any HTTP server on any box: WSL, the Pi, Ubuntu).

Everything technical is pulled LIVE from the code (prompt/config/tools import; config field docs
parsed from config.py's own comments), so the page cannot drift from the tree. Run verdicts and
narratives are curated in RUNS below — they are the session's judgment, not derivable from code.
"""

import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent / "site"
RUNS_DIR = ROOT / "scripts" / "dogfood" / "runs"

sys.path.insert(0, str(ROOT))

from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig  # noqa: E402
from shaderbox.copilot.prompt import _SYSTEM_PROMPT  # noqa: E402
from shaderbox.copilot.prompt_context import _CONVENTIONS  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from _caps import minimal_caps  # noqa: E402

from shaderbox.copilot.tools.registry import build_registry  # noqa: E402


def esc(s: str) -> str:
    return html.escape(str(s), quote=False)


def config_rows() -> list[tuple[str, str, str]]:
    # (name, value, doc) — docs parsed from the comment block right above each field in config.py.
    src = (ROOT / "shaderbox" / "copilot" / "config.py").read_text(encoding="utf-8")
    rows: list[tuple[str, str, str]] = []
    pending: list[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("# "))
            continue
        m = re.match(r"(\w+):\s*[\w\[\]\., |]+=\s*(.+)$", stripped)
        if m and m.group(1) in {f.name for f in fields(CopilotConfig)}:
            rows.append(
                (
                    m.group(1),
                    str(getattr(COPILOT_CONFIG, m.group(1))),
                    " ".join(pending),
                )
            )
            pending = []
        elif stripped and not stripped.startswith(
            ("class ", '"""', "from ", "import ")
        ):
            pending = []
    return rows


def tool_sections() -> str:
    reg = build_registry(minimal_caps())
    items = None
    for attr in ("_defs", "_definitions", "_tools", "_by_name"):
        v = getattr(reg, attr, None)
        if v:
            items = list(v.values()) if isinstance(v, dict) else list(v)
            break
    assert items
    out = []
    for tier, flag in (
        ("Eager — в каждом запросе", True),
        ("Lazy — подключаются через load_tools", False),
    ):
        rows = []
        for d in sorted((x for x in items if x.eager == flag), key=lambda x: x.name):
            schema = d.args_model.model_json_schema()
            props = schema.get("properties", {})
            req = set(schema.get("required", []))
            parts = []
            for n, p in props.items():
                t = p.get("type") or (
                    "|".join(a.get("type", "?") for a in p.get("anyOf", [])) or "any"
                )
                parts.append(
                    f"{n}: {t}"
                    + ("" if n in req else f" = {json.dumps(p.get('default'))}")
                )
            flags = []
            if d.mutating:
                flags.append("mutating")
            if d.is_edit:
                flags.append("edit-brake")
            if d.gate_policy.name != "NONE":
                flags.append(f"gate:{d.gate_policy.name}")
            rows.append(
                "<div class='tool'><code>{}({})</code> {}<p>{}</p></div>".format(
                    esc(d.name),
                    esc(", ".join(parts)),
                    "".join(f"<span class='chip'>{esc(f)}</span>" for f in flags),
                    esc((d.description or "").strip()),
                )
            )
        out.append(f"<h3>{esc(tier)} ({len(rows)})</h3>" + "\n".join(rows))
    return "\n".join(out)


def dialogue_html(project_dir: Path) -> str:
    conv = project_dir / "copilot" / "conversation.json"
    if not conv.exists():
        return "<p class='dim'>(диалог не сохранился)</p>"
    data = json.loads(conv.read_text(encoding="utf-8"))
    role_map = {
        "user": ("Юзер", "u"),
        "assistant": ("Копайлот", "a"),
        "error": ("Копайлот [движок]", "e"),
        "tool_status": ("[движок]", "t"),
    }
    out = []
    for m in data.get("messages", []):
        role, text = m.get("role"), (m.get("text") or "").strip()
        if role not in role_map or not text:
            continue
        label, cls = role_map[role]
        out.append(f"<div class='msg {cls}'><b>{label}:</b> {esc(text)}</div>")
    return "\n".join(out) or "<p class='dim'>(пусто)</p>"


def media_html(scenario_key: str, patterns: list[str]) -> str:
    dest = SITE / "media" / scenario_key
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for pat in patterns:
        for src in sorted(RUNS_DIR.glob(pat)):
            tgt = dest / src.name
            shutil.copy2(src, tgt)
            rel = f"media/{scenario_key}/{src.name}"
            if src.suffix == ".mp4":
                out.append(f"<video controls preload='metadata' src='{rel}'></video>")
            else:
                out.append(f"<a href='{rel}'><img src='{rel}' loading='lazy'></a>")
    return "\n".join(out) or "<p class='dim'>(медиа не сохранилось)</p>"


def md_lite(text: str) -> str:
    # Minimal markdown for the findings docs: headers, bold, code, tables, lists. Enough for our files.
    lines_out, in_table, in_list = [], False, False
    for raw in text.splitlines():
        line = esc(raw)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        if raw.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                lines_out.append("<table>")
                in_table = True
            lines_out.append(
                "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            )
            continue
        if in_table:
            lines_out.append("</table>")
            in_table = False
        if raw.startswith("- "):
            if not in_list:
                lines_out.append("<ul>")
                in_list = True
            lines_out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list and not raw.startswith(("  ", "- ")):
            lines_out.append("</ul>")
            in_list = False
        if raw.startswith("### "):
            lines_out.append(f"<h4>{line[4:]}</h4>")
        elif raw.startswith("## "):
            lines_out.append(f"<h3>{line[3:]}</h3>")
        elif raw.startswith("# "):
            lines_out.append(f"<h3>{line[2:]}</h3>")
        elif raw.strip():
            lines_out.append(f"<p>{line}</p>")
    if in_table:
        lines_out.append("</table>")
    if in_list:
        lines_out.append("</ul>")
    return "\n".join(lines_out)


# ---- Curated run manifest: the session's judgment (verdicts are human, not derivable) ----

RUNS: list[dict] = [
    dict(
        key="s03",
        num="03",
        title="Пять кругов (статика, чистый GLSL)",
        proj="proj-qqqszf1y",
        verdict="PASS, 3 сообщения",
        media=["proj-qqqszf1y/renders/*.png"],
        notes="Замерено: центр ряда y=199.5/200, шаг 75px... финал: маржи 14/15 ≈ зазоры 15px, Ø62-63. "
        "Слабость: «качель» раскладки — чинит одну констрейнту, ломая другую (2 корректировки). "
        "Sweep убрал мёртвые uniform'ы, поведение не тронул (пиксель-идентично).",
    ),
    dict(
        key="s04",
        num="04",
        title="Орбита (u_time, период 2с)",
        proj="proj-a2xdljvt",
        verdict="PASS с одного сообщения; период 2.000с (пиксельно); стабильность 3/3 в свипе",
        media=["proj-a2xdljvt/renders/*.mp4"],
        notes="Правильный выбор инструмента (без скрипта), aspect-коррекция применена, период через π.",
    ),
    dict(
        key="s05",
        num="05",
        title="Bounce (физика в script.py)",
        proj="proj-qogzoom0",
        verdict="PASS с одного go-ahead; свип 3/3 (класс «сквозь пол» ретрагирован — ошибка судьи)",
        media=["proj-qogzoom0/renders/*.mp4"],
        notes="Euler-интеграция, явный at_rest, restitution; траектория сверена численно: пики 0.357→0.169, покой.",
    ),
    dict(
        key="s08",
        num="08",
        title="Mixed grid 3×3 (компаунд из простых)",
        proj="proj-h_kod9m6",
        verdict="Ре-ран пост-058: 11/11 БЕЗ корректировок (пилот: 3 фейла)",
        media=["proj-h_kod9m6/renders/*.mp4"],
        notes="Сам решил y-flip (`2 - int(cell.y)`), вынес rotate()/PI, switch-диспетчер клеток. "
        "В стабилити-свипе компаунды дают 1-2 тайминг-слипа на прогон (блинк-рейт, замирание клетки).",
    ),
    dict(
        key="s09",
        num="09",
        title="Секундомер (неявные аффордансы, канвас 640×360)",
        proj="proj-u9zhg_9p",
        verdict="One-shot PASS: циферблат 1.000 аспект, штрихи 1.05px, период 60.0с — ни один uniform не назван",
        media=["proj-u9zhg_9p/renders/*.mp4"],
        notes="u_aspect/u_time/u_resolution применены сами, с гардом min(res)>=1. Дизайн сценария — мейнтейнера.",
    ),
    dict(
        key="s10",
        num="10",
        title="Pong (state-машина, AI-ракетки, счёт)",
        proj="proj-muddu21x",
        verdict="FAIL-at-budget* по счёту → после пост-бюджетного «Double the ball speed» счёт ожил (точка на ~40с)",
        media=["proj-muddu21x/renders/*.mp4"],
        notes="*одну корректировку сжёг судейский live-tick артефакт. Код-финдинг: dead-store правки "
        "(константа перезатирается _reset_ball) дважды заявлены как эффект → добавлен движковый "
        "value-no-op детект. Проводка счёта была корректна end-to-end с самого начала.",
    ),
    dict(
        key="s11",
        num="11",
        title="Кость 3D (raymarching, пипсы, тень)",
        proj="proj-gyw22jya",
        verdict="FAIL-at-budget по пипсам (2/2 прогонов); куб/вращение/свет/тень — есть",
        media=["proj-04dlzons/renders/*.mp4", "proj-gyw22jya/renders/*strip_t1-5*.png"],
        notes="После урока «локальные рамы» модель ВПЕРВЫЕ вырезает пипсы в правильной локальной системе — "
        "но зарывает их на 0.09 под поверхность (забыла, что sdRoundBox(b,r) раздувает бокс до b+r) "
        "и не может отладить с симптомов. Потолок дешёвой модели; триггер — сильная модель.",
    ),
    dict(
        key="s12",
        num="12",
        title="Радар (полярные координаты)",
        proj="proj-t_sol1wq",
        verdict="PASS с 1 корректировкой; развёртка ровно 4.0с, послесвечение направленное (+47% за лучом)",
        media=["proj-t_sol1wq/renders/*.mp4"],
        notes="Перпендикулярная ось тест-сета (полярка) — уроков не потребовала.",
    ),
    dict(
        key="s13",
        num="13",
        title="ФИНАЛЬНЫЙ ЭКЗАМЕН — пульт подлодки (все оси в одной сцене)",
        proj="proj-v4r16iqv",
        verdict="PASS по fidelity/motion/logic/honesty; process WEAK, code MIXED. 12 ходов, $0.60, 3 коррекции",
        media=[
            "proj-v4r16iqv/renders/*.mp4",
            "proj-v4r16iqv/renders/*strip_t1-14.5*.png",
        ],
        notes="Сонар (полярка, период РОВНО 4.0с, послесвечение позади луча, контакты вспыхивают при "
        "проходе) + 3D мина рейммарчем + шкала глубины со скриптовой state-машиной (50->300 при "
        "50 м/с, холд 3.0с, обратно, холд 3.0с — численно точно) на канвасе 800x450. "
        "Главная находка: строка фактов рендера — ГЛОБАЛЬНОЕ среднее по кадру, поэтому мелкий "
        "элемент (лампа, ~2% площади) для модели невидим: она починила лампу и три хода подряд "
        "честно писала «лампа всё ещё не видна». Плюс: ход 1 вывалил в чат свой черновик мышления "
        "(12k токенов, ноль тулов), а finalный sweep-ход УДАЛИЛ 7 строк, но ДОБАВИЛ две "
        "неиспользуемые функции.",
    ),
]

FINDINGS_DOCS = [
    (
        "Пилот cornerstone-сценариев + стабилити-свип + фиксы",
        "ai_docs/features/057_dogfood_axes_and_scenarios/02_rerun_post058.md",
    ),
    (
        "Эшелон-2 (pong, die3d) + промпт-обучение",
        "ai_docs/features/057_dogfood_axes_and_scenarios/03_echelon2.md",
    ),
    (
        "Финальный экзамен (пульт подлодки)",
        "ai_docs/features/057_dogfood_axes_and_scenarios/04_final_exam.md",
    ),
]

CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --dim:#767676; --line:#e3e3e3; --accent:#2563eb; --card:#f7f7f8; --mono:#0f172a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#101214; --fg:#e6e6e6; --dim:#9a9a9a; --line:#2a2e33; --accent:#7aa2ff; --card:#191c20; --mono:#dbe2ea; }
}
* { box-sizing:border-box; } body { margin:0; font:15px/1.55 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
.layout { display:flex; max-width:1200px; margin:0 auto; }
nav { position:sticky; top:0; align-self:flex-start; width:230px; padding:24px 12px; font-size:13px; height:100vh; overflow-y:auto; border-right:1px solid var(--line); }
nav a { display:block; color:var(--dim); text-decoration:none; padding:3px 8px; border-radius:6px; }
nav a:hover { color:var(--accent); background:var(--card); }
main { flex:1; min-width:0; padding:24px 32px 120px; }
h1 { font-size:26px; } h2 { font-size:20px; margin-top:44px; padding-top:12px; border-top:1px solid var(--line); }
h3 { font-size:16px; margin-top:26px; } h4 { font-size:14px; }
code { background:var(--card); padding:1px 5px; border-radius:5px; font-size:13px; }
pre { background:var(--card); color:var(--mono); padding:14px; border-radius:10px; overflow-x:auto; font-size:12.5px; line-height:1.45; }
table { border-collapse:collapse; margin:10px 0; font-size:13.5px; width:100%; }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
th { background:var(--card); }
.chip { display:inline-block; font-size:11px; padding:1px 7px; border:1px solid var(--line); border-radius:99px; color:var(--dim); margin-left:6px; }
.tool { padding:9px 0; border-bottom:1px dashed var(--line); } .tool p { margin:5px 0 0; color:var(--dim); font-size:13px; }
.msg { padding:8px 12px; margin:7px 0; border-radius:10px; background:var(--card); white-space:pre-wrap; }
.msg.u { border-left:3px solid var(--accent); } .msg.e { border-left:3px solid #d97706; }
.msg.t { color:var(--dim); font-size:13px; } .msg.a { border-left:3px solid var(--line); }
video,img { max-width:100%; border-radius:10px; margin:6px 0; display:block; }
.dim { color:var(--dim); } .verdict { font-weight:600; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--accent); }
.anchor { color:var(--dim); text-decoration:none; margin-right:6px; }
"""


def section(sid: str, title: str, body: str, toc: list[tuple[str, str]]) -> str:
    toc.append((sid, title))
    return (
        f"<h2 id='{sid}'><a class='anchor' href='#{sid}'>§</a>{esc(title)}</h2>\n{body}"
    )


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    toc: list[tuple[str, str]] = []
    parts: list[str] = []

    intro = (
        "<p>Полный срез копайлота ShaderBox: промпт, интерфейс, рамки движка и все тестовые прогоны "
        "с диалогами и медиа. Технические секции сняты с живого кода на момент генерации; вердикты "
        "прогонов — курируемая оценка сессии. Страница — хаб для комментариев: ссылайся на § секций.</p>"
        f"<p class='dim'>commit {esc(commit)} · модель прогонов: openai/gpt-5.1-codex-mini · "
        "сгенерировано scripts/agent_hub/generate.py</p>"
    )
    parts.append(section("intro", "О странице", intro, toc))

    cfg_rows = "".join(
        f"<tr><td><code>{esc(n)}</code></td><td><code>{esc(v)}</code></td><td>{esc(doc)}</td></tr>"
        for n, v, doc in config_rows()
    )
    ctx_model = (
        "<p>Промпт собирается из тиров по волатильности (стабильное выше — префикс кэшируется, ~4× дешевле): "
        "<b>STATIC</b> (системный промпт §3) → <b>RARE</b> (карта проекта: ноды/current/ошибки; каталог SB_*-библиотеки; "
        "каталог примеров; каталог lazy-тулов; conventions §4) → <b>DIALOGUE</b> (история: ТОЛЬКО натуральный язык — "
        "сообщения юзера + одно движковое summary на ход; исходники в историю не пишутся) → <b>PER_TURN</b> "
        "(сообщение юзера + WORKING SET: полный нумерованный исходник каждой ноды в работе, канвас, uniform'ы, "
        "ошибки, script.py — пересобирается каждый шаг, LRU-кап 6 членов с объявлением эвикций).</p>"
    )
    parts.append(
        section(
            "config",
            "Конфиг и рамки движка",
            ctx_model
            + f"<table><tr><th>ручка</th><th>значение</th><th>что делает</th></tr>{cfg_rows}</table>",
            toc,
        )
    )

    parts.append(
        section(
            "prompt",
            "Системный промпт (дословно)",
            f"<pre>{esc(_SYSTEM_PROMPT)}</pre>",
            toc,
        )
    )
    parts.append(
        section(
            "conventions",
            "Conventions-блок (RARE-тир, дословно)",
            f"<pre>{esc(_CONVENTIONS)}</pre>",
            toc,
        )
    )
    parts.append(section("tools", "Тулы (из реестра)", tool_sections(), toc))

    run_parts = []
    for r in RUNS:
        proj = RUNS_DIR / r["proj"]
        run_parts.append(
            f"<h3 id='{r['key']}'><a class='anchor' href='#{r['key']}'>§</a>{esc(r['num'])} — {esc(r['title'])}</h3>"
            f"<p class='verdict'>{esc(r['verdict'])}</p><p>{esc(r['notes'])}</p>"
            + media_html(r["key"], r["media"])
            + f"<details open><summary>Диалог</summary>{dialogue_html(proj)}</details>"
        )
    parts.append(
        section(
            "runs",
            "Тестовые прогоны: вердикты, медиа, диалоги",
            "\n".join(run_parts),
            toc,
        )
    )

    for title, rel in FINDINGS_DOCS:
        p = ROOT / rel
        if p.exists():
            parts.append(
                section(
                    re.sub(r"\W+", "-", rel),
                    f"Находки: {title}",
                    md_lite(p.read_text(encoding="utf-8")),
                    toc,
                )
            )

    nav = "\n".join(f"<a href='#{sid}'>{esc(t)}</a>" for sid, t in toc)
    page = (
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>ShaderBox Copilot — hub</title><style>{CSS}</style></head><body>"
        f"<div class='layout'><nav><b>Copilot hub</b>{nav}</nav><main><h1>ShaderBox Copilot — полный отчёт</h1>"
        + "\n".join(parts)
        + "</main></div></body></html>"
    )
    (SITE / "index.html").write_text(page, encoding="utf-8")
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site -> {SITE} ({total / 1e6:.1f} MB, {len(list(SITE.rglob('*')))} files)")


if __name__ == "__main__":
    main()
