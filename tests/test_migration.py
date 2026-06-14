"""The one-time 048 migration: pre-048 per-uniform `u_*.py` scripts are folded into the node's
`script.py` as a commented block, then trashed. Destructive + one-shot, so it gets a falsifiable
test. `migrate_per_uniform_scripts` + `detach_script` touch only `self.paths`, so they bind onto a
light stub with a real ProjectPaths (the test_script_driven_reject __get__ idiom) — no GL, no full
ProjectSession construction."""

import types
from pathlib import Path
from typing import Any

from shaderbox.paths import ProjectPaths
from shaderbox.project_session import ProjectSession


def _stub(project_dir: Path) -> Any:
    stub = types.SimpleNamespace(paths=ProjectPaths.for_root(project_dir))
    stub.detach_script = ProjectSession.detach_script.__get__(stub)
    stub.migrate_per_uniform_scripts = ProjectSession.migrate_per_uniform_scripts.__get__(
        stub
    )
    return stub


def _seed_orphans(nodes_dir: Path, node_id: str) -> Path:
    scripts = nodes_dir / node_id / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "u_a__float.py").write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx): return 0.5\n",
        encoding="utf-8",
    )
    (scripts / "u_b__vec3.py").write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx): return Vec3(1, 0, 0)\n",
        encoding="utf-8",
    )
    return scripts


def test_migration_folds_and_trashes(tmp_path: Path) -> None:
    stub = _stub(tmp_path)
    scripts = _seed_orphans(stub.paths.nodes_dir, "n0")

    stub.migrate_per_uniform_scripts()

    brain = (scripts / "script.py").read_text(encoding="utf-8")
    # Falsifier: each orphan's body must survive as a commented block (no silent loss).
    assert "# --- migrated from u_a__float.py" in brain
    assert "# --- migrated from u_b__vec3.py" in brain
    assert "#     def update(self, ctx): return 0.5" in brain
    # Falsifier: the orphans are gone from the live scripts dir (trashed, not left to drive nothing).
    assert not list(scripts.glob("u_*.py"))
    assert (stub.paths.trash_dir / "scripts" / "n0" / "u_a__float.py").is_file()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    stub = _stub(tmp_path)
    scripts = _seed_orphans(stub.paths.nodes_dir, "n0")

    stub.migrate_per_uniform_scripts()
    first = (scripts / "script.py").read_text(encoding="utf-8")
    # Falsifier: a second run finds no u_*.py and must NOT re-prepend (no double block).
    stub.migrate_per_uniform_scripts()
    second = (scripts / "script.py").read_text(encoding="utf-8")
    assert first == second
    assert second.count("# --- migrated from u_a__float.py") == 1


def test_migration_noop_without_orphans(tmp_path: Path) -> None:
    stub = _stub(tmp_path)
    node = stub.paths.nodes_dir / "n0"
    (node / "scripts").mkdir(parents=True)
    (node / "scripts" / "script.py").write_text("brain\n", encoding="utf-8")

    stub.migrate_per_uniform_scripts()
    # Falsifier: a node with only script.py (no orphans) is untouched.
    assert (node / "scripts" / "script.py").read_text(encoding="utf-8") == "brain\n"
