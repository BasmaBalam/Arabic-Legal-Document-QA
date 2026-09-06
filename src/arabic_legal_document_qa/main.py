from fastapi import FastAPI
from .routes import base, document


app = FastAPI()

@app.on_event("startup")
async def startup_spam():
    pass


@app.on_event("shutdown")
async def shutdown_spam():
    pass

app.include_router(base.base_router)
app.include_router(document.document_router)