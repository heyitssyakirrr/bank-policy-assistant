from fastapi import APIRouter

from app.core.db import check_db_connection

router = APIRouter()


@router.get("/health")
def health_check():
    db_ok = check_db_connection()
    return {"status": "ok" if db_ok else "degraded", "database": "connected" if db_ok else "unreachable"}