from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging import RequestContextMiddleware, setup_logging

setup_logging(settings.log_level)

app = FastAPI(title="Bank Policy Compliance Assistant")
app.add_middleware(RequestContextMiddleware)

app.include_router(chat.router)
app.include_router(health.router)

# Mounted last so it doesn't shadow the routes above
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")