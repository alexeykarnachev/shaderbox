"""The dogfooding station (feature 075): a durable, browsable home for every dogfooding
experiment. `report/` compiles an append-only JSONL log into a static site; `runs/` is the store
(one directory per experiment: its `events.jsonl` plus the media each attempt produced);
`index.html` is the one bookmark. The harness under `scripts/dogfood/` DRIVES a run; this package
RECORDS it and renders the record."""
