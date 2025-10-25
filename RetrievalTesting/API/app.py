from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from pydantic import BaseModel, Field, model_validator
import os
import io
import pandas as pd
import polars as pl
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedder import Retriever
from evaluator import analyze_all_cases

_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

# validating weights (set default + normalization)
class Weights(BaseModel):
    similarity: float = Field(0.3333333333, ge=0)
    court_stats: float = Field(0.3333333333, ge=0)
    citation: float = Field(0.3333333333, ge=0)

    @model_validator(mode="after")
    def normalize(self):
        total = float(self.similarity) + float(self.court_stats) + float(self.citation)
        if total <= 0:
            raise ValueError("At least one weight must be > 0.")
        # normalize so weights sum to 1
        self.similarity = float(self.similarity) / total
        self.court_stats = float(self.court_stats) / total
        self.citation = float(self.citation) / total
        return self

DEFAULT_WEIGHTS = Weights(similarity=0.333, court_stats=0.333, citation=0.334)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=200)
    # weights optional
    weights: Weights = Field(default_factory=Weights)

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    meta: Dict[str, Any]

class AnalysisResponse(BaseModel):
    analysis: str
    meta: Dict[str, Any]

# create api
app = FastAPI(title="Fair Use Legal Bot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
def health():
    return {"status": "ok"}

# retrieve documents
@app.post("/v1/retrieval", response_model=QueryResponse)
@limiter.limit("15/minute")
def search_similar_cases(
    request: Request,
    req: QueryRequest,
    format: str = Query("json", pattern="^(json|csv)$"), # use ?format=json|csv
):
    try:
        w = req.weights
        s, cstats, cit = float(w.similarity), float(w.court_stats), float(w.citation)

        df = get_retriever().search_similar_cases(
            req.query,
            similarity_weight=s,
            court_weight=cstats,
            case_weight=cit,
            top_k=req.top_k,
        )

        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        elif hasattr(df, "to_pandas"):
            df = df.to_pandas()

        if not isinstance(df, pd.DataFrame):
            t = f"{type(df).__module__}.{type(df).__name__}"
            raise HTTPException(
                status_code=500,
                detail=f"Retriever did not return a pandas.DataFrame (got {t})"
            )

        if format == "csv":
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="results.csv"'},
            )

        records = df.to_dict(orient="records")
        return JSONResponse({
            "results": records,
            "meta": {
                "query": req.query,
                "top_k": req.top_k,
                "weights": {"similarity": s, "court_stats": cstats, "citation": cit},
                "rows": len(records),
                "weights_normalized": True,
            },
        })

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# conduct full analysis
@app.post("/v1/analysis", response_model=AnalysisResponse)
@limiter.limit("15/minute")
def analyze(
    request: Request,
    req: QueryRequest,
):
    try:
        w = req.weights
        s, cstats, cit = float(w.similarity), float(w.court_stats), float(w.citation)

        df = get_retriever().search_similar_cases(
            req.query,
            similarity_weight=s,
            court_weight=cstats,
            case_weight=cit,
            top_k=req.top_k,
        )

        analysis = analyze_all_cases(df, req.query)
        
        return JSONResponse({
            "results": analysis,
            "meta": {
                "query": req.query,
                "top_k": req.top_k,
                "weights": {"similarity": s, "court_stats": cstats, "citation": cit},
                "rows": len(analysis),
                "weights_normalized": True,
            },
        })
    
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))