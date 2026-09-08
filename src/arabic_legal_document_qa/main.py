from fastapi import FastAPI
from .routes import base, document
from .stores.llm.LLMProviderFactory import LLMProviderFactory
from .configs.config import Settings, get_settings

app = FastAPI()

@app.on_event("startup")
async def startup_spam():
    
    settings = get_settings()
    llm_provider_factory = LLMProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider = settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id= settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider = settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)


@app.on_event("shutdown")
async def shutdown_spam():
    pass

app.include_router(base.base_router)
app.include_router(document.document_router)