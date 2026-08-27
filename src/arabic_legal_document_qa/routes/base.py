from fastapi import FastAPI, APIRouter,Depends
from ..configs.config import Settings, get_settings

base_router = APIRouter(
    prefix = "/api/v1",
    tags = ["system"]
)

@base_router.get("/")
async def welcom(app_settings:Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "App Name": app_name,
        "App Version": app_version,
    }