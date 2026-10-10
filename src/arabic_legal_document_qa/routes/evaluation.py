import logging
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from arabic_legal_document_qa.models.enums import ResponseStatus
from arabic_legal_document_qa.controllers.RAGController import RAGController
from langchain_ollama import ChatOllama, OllamaEmbeddings
from arabic_legal_document_qa.routes.schemes.evaluation import EvaluationRequest
from arabic_legal_document_qa.controllers.EvaluationController import (
    EvaluationController,
)

logger = logging.getLogger(__name__)


evaluation_router = APIRouter(prefix="/api/v1", tags=["Evaluation"])


@evaluation_router.post(
    "/evaluation", status_code=status.HTTP_200_OK, summary="Evaluate the RAG Pipeline"
)
def run_evaluation(request: Request, evaluation_request: EvaluationRequest):
    """Evaluate the RAG pipeline using RAGAS and log experiment results to MLflow."""
    settings = request.app.state.settings
    rag_controller = RAGController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    # Configure the LLM used to evaluate generated answers.
    evaluator_llm = ChatOllama(
        model=settings.EVALUATION_MODEL_ID,
        temperature=0,
        format="json",
        reasoning=False,
        base_url=settings.OLLAMA_HOST,
    )

    # Configure the embeddings used by the evaluation pipeline.
    evaluator_embeddings = OllamaEmbeddings(
        model=settings.EVALUATION_EMBEDDING_MODEL_ID,
        base_url=settings.OLLAMA_HOST,
    )

    evaluation_controller = EvaluationController(
        mlflow_experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
        rag_controller=rag_controller,
        evaluation_llm=evaluator_llm,
        evaluation_embeddings=evaluator_embeddings,
    )

    experiment_params = {
        # RAG configuration
        "rag_embedding_backend": settings.EMBEDDING_BACKEND,
        "rag_embedding_model": settings.EMBEDDING_MODEL_ID,
        "rag_embedding_dimension": settings.EMBEDDING_MODEL_SIZE,
        "rag_generation_backend": settings.GENERATION_BACKEND,
        "rag_generation_model": settings.GENERATION_MODEL_ID,
        # Chunking configuration
        "chunk_size": evaluation_request.chunk_size,
        "overlap_size": evaluation_request.overlap_size,
        # Generation configuration
        "generation_max_input_characters": settings.DEFAULT_INPUT_MAX_CHARACTERS,
        "generation_max_output_tokens": settings.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
        "generation_temperature": settings.DEFAULT_GENERATION_TEMPERATURE,
    }

    try:
        evaluation_data = evaluation_controller.load_evaluation_data("cases_v1.jsonl")

        # Run the evaluation and extract the resulting metrics.
        result = evaluation_controller.evaluate(
            evaluation_data=evaluation_data,
            experiment_params=experiment_params,
        )

        scores = evaluation_controller.extract_scores(result)
        # Determine whether the evaluation metrics meet the configured thresholds.
        evaluation_status = evaluation_controller.apply_evaluation_thresholds(scores)

        return JSONResponse(
            content={
                "signal": ResponseStatus.EVALUATION_SUCCESS.value,
                "status": evaluation_status,
                "scores": scores,
                "questions_evaluated": len(evaluation_data),
            }
        )

    except Exception as exc:
        logger.exception("Failed to run RAGAS evaluation")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseStatus.EVALUATION_FAILED.value,
                "detail": "Evaluation failed. Check the application logs.",
            },
        )
