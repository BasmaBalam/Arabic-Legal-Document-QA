from fastapi import FastAPI, APIRouter,Request
from arabic_legal_document_qa.configs.config import Settings, get_settings
from rag import AskRequest

rag_router = APIRouter(
    prefix = "/api/v1",
    tags = ["system"]
)

@rag_router.post("/ask")
async def ask_endpoint(request: Request, ask_request: AskRequest):
    pass