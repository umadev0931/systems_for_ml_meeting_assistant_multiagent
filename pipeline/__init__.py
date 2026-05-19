"""Pipeline package: sequential, parallel, langgraph topologies."""
from pipeline.base import PipelineResult, AgentBundle
from pipeline.sequential import run_sequential
from pipeline.parallel import run_parallel
from pipeline.langgraph_pipeline import run_langgraph

# mode name -> coroutine(transcript, size, client, tracer, stream)
PIPELINES = {
    "sequential": run_sequential,
    "parallel": run_parallel,
    "langgraph": run_langgraph,
}

__all__ = ["PipelineResult", "AgentBundle", "PIPELINES",
           "run_sequential", "run_parallel", "run_langgraph"]
