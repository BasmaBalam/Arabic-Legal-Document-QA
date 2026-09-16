import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from arabic_legal_document_qa.controllers.RAGController import RAGController
from arabic_legal_document_qa.models.enums import ResponseStatus
from arabic_legal_document_qa.routes.schemes.document import IndexRequest

logger = logging.getLogger(__name__)

document_router = APIRouter(prefix="/api/v1", tags=["system"])


@document_router.post(
    "/index",
    status_code=status.HTTP_200_OK,
    summary="Chunk and Index Legal Documents into Vector DB",
)
async def index_documents(request: Request, index_request: IndexRequest):
    """Chunk legal documents and index their embeddings into the vector database."""

    logger.info("Starting legal document indexing")
    rag_controller = RAGController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    # Chunk documents using DocumentController
    chunks = rag_controller.chunk_documents(
        chunk_size=index_request.chunk_size, overlap_size=index_request.overlap_size
    )

    if not chunks:
        logger.error("Failed to chunk legal document")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseStatus.CHUNKING_FAILED.value},
        )

    logger.info(
        "Legal document chunking completed successfully: %d chunks created", len(chunks)
    )

    # Embed & Index chunks into Qdrant
    indexed_count = rag_controller.index_documents(chunks, index_request.do_reset)

    if indexed_count == 0:
        logger.error("Failed to index document chunks into vector database")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseStatus.INDEXING_FAILED.value},
        )

    logger.info(
        "Legal document indexing completed successfully: %d chunks indexed",
        indexed_count,
    )
    return JSONResponse(
        content={
            "signal": ResponseStatus.INDEXING_SUCCESS.value,
            "indexed_count": indexed_count,
        }
    )
