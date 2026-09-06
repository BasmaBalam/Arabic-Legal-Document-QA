from fastapi import FastAPI, APIRouter,Request
from fastapi.responses import JSONResponse
from arabic_legal_document_qa.models.enums import ResponseStatus
from arabic_legal_document_qa.configs.config import Settings, get_settings
from arabic_legal_document_qa.routes.schemes.document import ProcessRequest
from arabic_legal_document_qa.controllers import DocumentController

document_router = APIRouter(
    prefix = "/api/v1",
    tags = ["system"]
)


@document_router.post("/process")
async def process_document_endpoint(request: Request, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    document_controller = DocumentController()
    
    doc_content = document_controller.get_document_content()

    chunks = document_controller.process_document_content(
        doc_content= doc_content,
        chunk_size= chunk_size,
        overlap_size= overlap_size
    )

    if chunks is None or len(chunks) == 0:
        return JSONResponse(
            status_code= status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseStatus.PROCESSING_FAILED.value
                }
        )

    return JSONResponse(
        content={
            "signal": ResponseStatus.PROCESSING_SUCCESS.value,
            "chunk_count": len(chunks),
            "article_count": len(doc_content),
        }
    )