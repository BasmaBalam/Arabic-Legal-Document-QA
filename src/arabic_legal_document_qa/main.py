import uuid

from fastapi import FastAPI, Request

from .configs.config import get_settings
from .configs.logging_config import correlation_id_ctx, setup_logging
from .routes import ask, base, document, evaluation
from .stores.llm.LLMProviderFactory import LLMProviderFactory
from .stores.llm.templates.template_parser import TemplateParser
from .stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

setup_logging()
app = FastAPI()


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    correlation_id_ctx.set(correlation_id)

    response = await call_next(request)

    response.headers["X-Request-ID"] = correlation_id
    return response


@app.on_event("startup")
async def startup_spam():
    settings = get_settings()
    app.state.settings = settings
    llm_provider_factory = LLMProviderFactory(settings)
    vector_provider_factory = VectorDBProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(
        provider=settings.GENERATION_BACKEND
    )
    # app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
    generation_model_id = (
        settings.VLLM_MODEL_ID
        if settings.GENERATION_BACKEND == "VLLM"
        else settings.GENERATION_MODEL_ID
    )

    app.generation_client.set_generation_model(
        model_id=generation_model_id
    )

    # embedding client
    app.embedding_client = llm_provider_factory.create(
        provider=settings.EMBEDDING_BACKEND
    )
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    # vector db client
    app.vectordb_client = vector_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    app.template_parser = TemplateParser(language=settings.DEFAULT_LANG)


@app.on_event("shutdown")
async def shutdown_spam():
    app.vectordb_client.disconnect()


app.include_router(base.base_router)
app.include_router(document.document_router)
app.include_router(ask.ask_router)
app.include_router(evaluation.evaluation_router)
