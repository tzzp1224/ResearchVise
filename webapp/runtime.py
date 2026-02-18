"""Shared runtime singletons for v2 web/CLI entrypoints."""

from __future__ import annotations

from orchestrator.queue import InMemoryRunQueue
from orchestrator.service import RunOrchestrator
from orchestrator.store import InMemoryRunStore
from pipeline_v2.runtime import RunPipelineRuntime
from render.manager import RenderManager


_STORE = InMemoryRunStore()
_QUEUE = InMemoryRunQueue()
_ORCHESTRATOR = RunOrchestrator(store=_STORE, queue=_QUEUE)
_RENDER_MANAGER = RenderManager()
_RUNTIME = RunPipelineRuntime(orchestrator=_ORCHESTRATOR, render_manager=_RENDER_MANAGER)


def get_orchestrator() -> RunOrchestrator:
    return _ORCHESTRATOR


def get_runtime() -> RunPipelineRuntime:
    return _RUNTIME
