"""POST /turns/{id}/feedback: thumbs up or down on an answer (a product
signal for finding bad answers, not model training)."""

import uuid

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import ServicesDep
from app.api.schemas import FeedbackRequest

router = APIRouter()


@router.post("/turns/{turn_id}/feedback", status_code=204)
def add_feedback(turn_id: uuid.UUID, feedback: FeedbackRequest, services: ServicesDep) -> Response:
    if not services.audit.add_feedback(turn_id, feedback.rating, feedback.comment):
        raise HTTPException(status_code=404, detail="turn not found")
    return Response(status_code=204)
