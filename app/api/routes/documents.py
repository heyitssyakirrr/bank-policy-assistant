"""The three policy PDFs, so a citation can open the PDF at its page."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.agent.schemas import DOCUMENT_TITLES
from app.api.schemas import DocumentInfo
from app.api.views import pdf_path
from app.core.config import settings

router = APIRouter()

DOCUMENT_FILES = {
    "ekyc": "pd-ekyc-apr2024.pdf",
    "ftfc": "pd-ftfc-mar24.pdf",
    "amlcft": "PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf",
}


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(code=code, title=DOCUMENT_TITLES[code], pdf_url=pdf_path(code)) for code in DOCUMENT_FILES]


@router.get("/documents/{code}.pdf")
def document_pdf(code: str) -> FileResponse:
    if code not in DOCUMENT_FILES:
        raise HTTPException(status_code=404, detail="unknown document")
    # inline, so "#page=N" opens the browser's PDF viewer at the cited page
    return FileResponse(
        settings.raw_pdf_dir / DOCUMENT_FILES[code], media_type="application/pdf", content_disposition_type="inline"
    )
