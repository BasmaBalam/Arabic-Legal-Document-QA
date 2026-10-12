import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from arabic_legal_document_qa.controllers.RAGController import RAGController
from arabic_legal_document_qa.models.enums import ResponseStatus
from arabic_legal_document_qa.routes.schemes.health import HealthResponse

logger = logging.getLogger(__name__)

base_router = APIRouter(prefix="/api/v1", tags=["system"])


@base_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health Check and Collection Stats",
)
async def health_check(request: Request):
    """Check application health and return the indexed document count."""

    logger.info("Starting health check")
    try:
        rag_controller = RAGController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
        )

        indexed_count = rag_controller.get_indexed_count()

        health_data = HealthResponse(status="healthy", documents_indexed=indexed_count)

        logger.info(
            "Health check completed successfully: %d documents indexed", indexed_count
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseStatus.HEALTH_SUCCESS.value,
                "health_response": health_data.model_dump(),
            },
        )

    except Exception:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseStatus.HEALTH_FAILED.value},
        )
