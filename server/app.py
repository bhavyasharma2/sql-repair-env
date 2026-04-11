"""
server/app.py — FastAPI application for SQL Repair Env.

Uses the official OpenEnv create_app helper so the environment
automatically gets:
  - POST /reset
  - POST /step
  - GET  /state
  - GET  /health
  - GET  /schema
  - GET  /metadata
  - WebSocket /ws
  - Web UI at /web (when ENABLE_WEB_INTERFACE=1)
"""

import uvicorn

try:
    from openenv.core.env_server import create_app
    from ..models import SQLRepairAction, SQLRepairObservation
    from .sql_repair_environment import SQLRepairEnvironment
except ImportError:
    from openenv.core.env_server import create_app
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SQLRepairAction, SQLRepairObservation
    from server.sql_repair_environment import SQLRepairEnvironment

from fastapi.responses import RedirectResponse

app = create_app(
    SQLRepairEnvironment,
    SQLRepairAction,
    SQLRepairObservation,
    env_name="sql-repair-env",
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
