import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from arabic_legal_document_qa.controllers.RAGController import RAGController
from arabic_legal_document_qa.models.enums import ResponseStatus
from arabic_legal_document_qa.routes.schemes.ask import AskRequest

logger = logging.getLogger(__name__)

ask_router = APIRouter(prefix="/api/v1", tags=["Ask"])


@ask_router.post(
    "/ask", status_code=status.HTTP_200_OK, summary="Ask a Question using RAG Pipeline"
)
def ask(request: Request, ask_request: AskRequest):
    """Answer a legal question using the RAG pipeline."""

    logger.info("Starting RAG question answering")
    rag_controller = RAGController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, sources, _contexts = rag_controller.answer(
        question=ask_request.question, limit=ask_request.limit
    )

    if not answer:
        logger.error("Failed to generate answer using RAG pipeline")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseStatus.ANSWERING_FAILED.value},
        )

    logger.info(
        "RAG question answered successfully: %d sources retrieved",
        len(sources) if sources else 0,
    )
    return JSONResponse(
        content={
            "signal": ResponseStatus.ANSWERING_SUCCESS.value,
            "answer": answer,
            "sources": sources,
        }
    )
