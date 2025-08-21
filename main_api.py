from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Processing & Ambiguity Detection API",
    description="API to process JSON data and flag ambiguous rows without C-extensions",
)

# Request models
class JsonDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    metadata: Optional[Dict[str, Any]] = None

class AbapDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    source_info: Optional[Dict[str, Any]] = None
    table_info: Optional[Dict[str, Any]] = None

class DataProcessor:
    """Processes data and flags ambiguous rows using Z-score (pure Python)"""

    def clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df.select_dtypes(include=[np.number]).fillna(0)
        if df_num.empty:
            raise ValueError("No numeric columns for ambiguity detection")
        return df_num

    def flag_ambiguity(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = self.clean_numeric(df)
        # Compute Z-scores
        means = df_num.mean()
        stds = df_num.std(ddof=0).replace(0, np.nan).fillna(1)
        zscores = (df_num - means) / stds
        # Flag any row with any |Z| > 3 as ambiguous
        df["ambiguity_flag"] = np.where(zscores.abs().max(axis=1) > 3, "ambiguous", "ok")
        return df

    def process(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        if not data:
            raise ValueError("No data provided")
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        return self.flag_ambiguity(df)

processor = DataProcessor()

@app.get("/")
async def root():
    return {
        "message": "Ambiguity Detection API is running",
        "endpoints": {
            "1": "POST /process-user-data",
            "2": "POST /process-abap-data"
        },
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process-user-data")
async def process_user_data(request: JsonDataRequest):
    try:
        df = processor.process(request.data)
        records = df.to_dict("records")
        summary = dict(df["ambiguity_flag"].value_counts())
        response = {
            "status": "success",
            "source": "user_json",
            "ambiguity_summary": summary,
            "processed_data": records,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
        }
        if request.metadata:
            response["metadata"] = request.metadata
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"User data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-abap-data")
async def process_abap_data(request: AbapDataRequest):
    try:
        df = processor.process(request.data)
        records = df.to_dict("records")
        summary = dict(df["ambiguity_flag"].value_counts())
        response = {
            "status": "success",
            "source": "abap_system",
            "ambiguity_summary": summary,
            "processed_data": records,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
        }
        if request.source_info:
            response["source_info"] = request.source_info
        if request.table_info:
            response["table_info"] = request.table_info
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"ABAP data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
