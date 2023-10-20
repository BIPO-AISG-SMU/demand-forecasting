import logging
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from bipo_fastapi.config import SETTINGS
from bipo_fastapi.v1.routers import model

# Set up logging
LOGGER = logging.getLogger("kedro")
LOGGER.info("Setting up logging configuration.")

# FastAPI application setup
version = SETTINGS.API_VERSION
app = FastAPI(title=SETTINGS.API_NAME, openapi_url=f"{version}/openapi.json")
api_router = APIRouter()
api_router.include_router(model.router, prefix="/model", tags=["model"])
app.include_router(api_router, prefix=version)

# In production mode,  specify explicitly the allowed origins for CORS
# In pre-production mode, use "*" wildcard for origin
ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
