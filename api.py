from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json, asyncio

from app.orchestrator import run_chain
from app.schemas import ValuationResponse

# ================= FastAPI app =================
app = FastAPI(title="Ag IQ v2 – Agent Edition")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test endpoint to verify API is working
@app.get("/api/status")
def api_status():
    return JSONResponse({"status": "active", "message": "Farm Equipment Valuation API is running"})

# serve static UI
static_path = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
def root():
    return FileResponse(static_path / "index.html")

# request / response models
class ValuationRequest(BaseModel):
    make: str
    model: str
    year: int
    condition: str
    description: str

@app.post("/v2/value", response_model=ValuationResponse)
async def value(req: ValuationRequest):
    try:
        result_json = await run_chain(req.model_dump())
        # result_json is already schema-validated by Agent-3
        return ValuationResponse.model_validate_json(result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
