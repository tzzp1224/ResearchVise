"""CLI entrypoint for v2 orchestration + runtime workers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

from core import RunMode, RunRequest
from webapp.runtime import get_orchestrator, get_runtime


def _json(text: str):
    raw = str(text or "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="AcademicResearchAgent v2 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ond = sub.add_parser("ondemand")
    ond.add_argument("--user-id", required=True)
    ond.add_argument("--topic", required=True)
    ond.add_argument("--time-window", default="24h")
    ond.add_argument("--tz", default="UTC")
    ond.add_argument("--budget-json", default="{}")
    ond.add_argument("--targets", default="web,mp4")

    daily = sub.add_parser("daily-subscribe")
    daily.add_argument("--user-id", required=True)
    daily.add_argument("--run-at", default="08:00")
    daily.add_argument("--tz", default="UTC")
    daily.add_argument("--top-k", type=int, default=3)

    tick = sub.add_parser("daily-tick")
    tick.add_argument("--now-utc", default="")

    sub.add_parser("worker-run-next")
    sub.add_parser("worker-render-next")

    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)

    cancel = sub.add_parser("cancel")
    cancel.add_argument("--run-id", required=True)

    args = parser.parse_args()
    orchestrator = get_orchestrator()
    runtime = get_runtime()

    if args.command == "ondemand":
        budget = _json(args.budget_json)
        targets = [item.strip() for item in str(args.targets).split(",") if item.strip()]
        req = RunRequest(
            user_id=args.user_id,
            mode=RunMode.ONDEMAND,
            topic=args.topic,
            time_window=args.time_window,
            tz=args.tz,
            budget=budget,
            output_targets=targets,
        )
        idem = f"ondemand:{args.user_id}:{args.topic}:{args.time_window}:{args.tz}"
        run_id = orchestrator.enqueue_run(req, idempotency_key=idem)
        print(json.dumps({"run_id": run_id}, ensure_ascii=False))
        return

    if args.command == "daily-subscribe":
        sub_id = orchestrator.schedule_daily_digest(
            user_id=args.user_id,
            run_at=args.run_at,
            tz=args.tz,
            top_k=int(args.top_k),
        )
        print(json.dumps({"subscription_id": sub_id}, ensure_ascii=False))
        return

    if args.command == "daily-tick":
        now_utc = None
        if str(args.now_utc).strip():
            now_utc = datetime.fromisoformat(str(args.now_utc).replace("Z", "+00:00")).astimezone(timezone.utc)
        run_ids = orchestrator.trigger_due_daily_runs(now_utc=now_utc)
        print(json.dumps({"created_run_ids": run_ids}, ensure_ascii=False))
        return

    if args.command == "worker-run-next":
        result = runtime.run_next()
        if not result:
            print(json.dumps({"processed": False}, ensure_ascii=False))
            return
        print(
            json.dumps(
                {
                    "processed": True,
                    "run_id": result.run_id,
                    "output_dir": result.output_dir,
                    "render_job_id": result.render_job_id,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "worker-render-next":
        status = runtime.process_next_render()
        if not status:
            print(json.dumps({"processed": False}, ensure_ascii=False))
            return
        print(
            json.dumps(
                {
                    "processed": True,
                    "run_id": status.run_id,
                    "render_job_id": status.render_job_id,
                    "state": status.state,
                    "output_path": status.output_path,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "status":
        print(json.dumps(runtime.get_run_bundle(args.run_id), ensure_ascii=False, default=str))
        return

    if args.command == "cancel":
        canceled = orchestrator.cancel_run(args.run_id)
        status_payload = orchestrator.get_run_status(args.run_id)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "canceled": canceled,
                    "status": status_payload.model_dump(mode="json") if status_payload else None,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
