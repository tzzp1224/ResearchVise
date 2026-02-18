# AcademicResearchAgent v2

This repository now runs a unified v2 pipeline:

`RunRequest -> connectors -> normalize -> dedup/cluster -> rank -> script -> storyboard -> prompt compile -> render queue -> postprocess -> export/notify`

## Entry Points

- CLI: `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/main.py`
- API: `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/webapp/v2_app.py`
- Smoke command: `/Users/dexter/Documents/Dexter_Work/AcademicResearchAgent/scripts/e2e_smoke_v2.py`

## Quick Start

1. On-demand enqueue

```bash
python main.py ondemand --user-id u1 --topic "mcp deployment" --time-window 24h --tz America/Los_Angeles
```

2. Process run worker

```bash
python main.py worker-run-next
```

3. Process render worker

```bash
python main.py worker-render-next
```

4. Query status

```bash
python main.py status --run-id <run_id>
```

## Daily Digest

1. Register schedule (default 08:00 local)

```bash
python main.py daily-subscribe --user-id u1 --tz America/Los_Angeles --run-at 08:00 --top-k 3
```

2. Trigger due schedules (for cron/timer worker)

```bash
python main.py daily-tick
```

## Local Smoke Test

```bash
python scripts/e2e_smoke_v2.py --out-dir /tmp/ara_v2_smoke
```

The command prints a JSON bundle containing run status and artifact paths.
