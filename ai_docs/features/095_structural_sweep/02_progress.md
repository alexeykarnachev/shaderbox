# 095 — Structural sweep: the log

What actually happened, wave by wave, appended as each lands. A log, not a plan — the plan is
`01_spec.md`. **On resume, read this file and `git log --oneline` first**: they are the truth
about where the work stopped.

Each entry carries its done-condition (written before the wave started), the verification
result, what was ruled out and why, and any surprise worth the next reader's time.

## W-0 inventory — NOT STARTED

done-condition (written in advance): every kind of Python symbol enumerated across every
directory named in the spec's wave list, sorted into the SAFE / CAREFUL / RISKY tiers, with the
enumerating command recorded here so the next session can re-run it rather than trust the
result. No file in `shaderbox/`, `scripts/` or `tests/` is modified by this wave.
