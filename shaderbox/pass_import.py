"""The import plan: what copying another document's passes into this one changes (091 D4).

Pure and GL-free, importing `pass_graph` only. The session executes the plan; this decides it,
so every rejection and every rewired sampler is testable with two wirings and no context.

A copied pass keeps its shader text; what changes is its NAME (the group name prefixed) and the
stored SOURCE of every sampler the source wiring filled, written explicitly under the new
names because the name rule (`u_blur` reads `blur`) does not survive the prefix. An entry point
the host feeds is not copied: its readers point at the host pass instead, and the host's own
readers of that pass are handed to the bundle's output (D6).
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from shaderbox.pass_graph import PASS_NAME_RE, Wiring, entry_points, plan_passes


@dataclass(frozen=True)
class ImportPlan:
    """Every write `import_passes` makes, decided before any of them happens."""

    # source name -> host name, for every COPIED pass
    renames: dict[str, str]
    # host name -> sampler -> the host pass it reads, for every sampler the source wiring filled
    sources: dict[str, dict[str, str]]
    # the bundle's output under its host name
    output: str
    # HOST pass -> sampler -> `output`: the readers handed over (D6)
    handovers: dict[str, dict[str, str]] = field(default_factory=dict)
    # the document's output moves to `output` (D6)
    becomes_output: bool = False


def plan_import(
    source_wiring: Wiring,
    source_output: str,
    group: str,
    substitutions: Mapping[str, str],
    handovers: Collection[tuple[str, str]],
    host_wiring: Wiring,
    host_output: str,
) -> ImportPlan | str:
    """The plan, or the message that rejects it.

    `substitutions` maps an entry point of the source to the host pass that feeds it;
    `handovers` are the `(host pass, sampler)` pairs that will read the bundle's output
    instead of the pass they read now. Every name is checked against the two wirings, so a
    plan that comes back can be executed without a further question.
    """
    if group and not PASS_NAME_RE.match(group):
        return "a group name starts with a letter and holds letters, digits and underscores"
    host_names = set(host_wiring)
    roots = set(entry_points(source_wiring))
    for entry, host in substitutions.items():
        if entry not in roots:
            return f"'{entry}' is not an entry point"
        if host not in host_names:
            return f"no such pass '{host}'"
    if source_output in substitutions:
        return f"'{source_output}' is the output and stays"
    copied = [name for name in sorted(source_wiring) if name not in substitutions]
    if not copied:
        return "nothing to import: every pass is replaced"

    prefix = f"{group}_" if group else ""
    renames = {name: f"{prefix}{name}" for name in copied}
    taken = sorted(new for new in renames.values() if new in host_names)
    if taken:
        return (
            f"{', '.join(repr(n) for n in taken)} already exist"
            if len(taken) > 1
            else (f"'{taken[0]}' already exists")
        )
    output = renames.get(source_output, renames[copied[0]])

    sources: dict[str, dict[str, str]] = {}
    for name in copied:
        rows: dict[str, str] = {}
        for uniform, read in source_wiring[name].items():
            if read in substitutions:
                rows[uniform] = substitutions[read]
            elif read in renames:
                rows[uniform] = renames[read]
        if rows:
            sources[renames[name]] = rows

    fed = set(substitutions.values())
    handed: dict[str, dict[str, str]] = {}
    for host_pass, uniform in handovers:
        if host_wiring.get(host_pass, {}).get(uniform) not in fed:
            return f"'{host_pass}.{uniform}' does not read a replaced pass"
        handed.setdefault(host_pass, {})[uniform] = output

    # The wiring the import would leave: a handover onto a host pass that itself feeds the
    # bundle closes a loop, and nothing downstream reports one loudly.
    merged: dict[str, dict[str, str]] = {
        name: dict(reads) for name, reads in host_wiring.items()
    }
    for host_pass, rows in handed.items():
        merged[host_pass].update(rows)
    for name in copied:
        merged[renames[name]] = dict(sources.get(renames[name], {}))
    _, errors = plan_passes(merged)
    if errors:
        return f"a loop through '{errors[0].pass_name}': uncheck a handover"

    return ImportPlan(
        renames=renames,
        sources=sources,
        output=output,
        handovers=handed,
        becomes_output=host_output in fed,
    )
