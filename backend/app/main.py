"""AgentDS backend entrypoint.

A plain FastAPI app: no graph-orchestration framework, no database. Agents will
be added later as plain Python classes called directly from routers.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load agentds/backend/.env into the process environment before anything reads it.
# Without this, `uvicorn app.main:app` would not see ANTHROPIC_API_KEY / AGENTDS_DATA_DIR
# from the .env file. Existing real environment variables always win (override=False).
load_dotenv()

from app.routers import (  # noqa: E402  (must follow load_dotenv above)
    analyze,
    clean,
    datasets,
    explain,
    recommend,
    report,
    status,
    train,
    visualize,
)

app = FastAPI(title="AgentDS", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(analyze.router)
app.include_router(clean.router)
app.include_router(recommend.router)
app.include_router(visualize.router)
app.include_router(train.router)
app.include_router(explain.router)
app.include_router(report.router)
app.include_router(status.router)


@app.get("/")
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
