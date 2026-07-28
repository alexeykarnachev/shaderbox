# Agent hub

Статическая страница-отчёт по копайлоту (промпт, тулы, конфиг, все дог-фуд прогоны с диалогами и медиа).

```sh
uv run python scripts/agent_hub/generate.py            # собрать site/ (медиа берёт из scripts/dogfood/runs/)
python3 -m http.server 8321 --directory scripts/agent_hub/site
```

Открыть: `http://<ip-этой-машины>:8321/`. Всё self-contained, никаких зависимостей кроме репо.
