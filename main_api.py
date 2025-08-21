from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Processing & Ambiguity Detection API",
    description="API to process JSON data from users or ABAP and flag ambiguous (anomalous) rows",
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
    """Processes data and flags ambiguous (anomalous) rows via IsolationForest"""

    def __init__(self):
        # You could preload/train this on historical data; here we train per-request
        self.model = IsolationForest(contamination=0.05, random_state=42)

    def clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric columns, filling NaNs with 0."""
        df_num = df.select_dtypes(include=[np.number])
        if df_num.empty:
            raise ValueError("No numeric columns for ambiguity detection")
        return df_num.fillna(0)

    def flag_ambiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model to this batch and predict anomalies:
        -1 → ambiguous/anomalous, 1 → normal
        """
        df_num = self.clean_numeric(df)
        self.model.fit(df_num)
        preds = self.model.predict(df_num)
        df["ambiguity_flag"] = np.where(preds == -1, "ambiguous", "ok")
        return df

    def process(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """Core: convert to DataFrame and flag ambiguous rows."""
        if not data:
            raise ValueError("No data provided")
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records")
        return self.flag_ambiguous(df)

processor = DataProcessor()

@app.get("/")
async def root():
    return {
        "message": "Data Processing & Ambiguity Detection API is running",
        "endpoints": {
            "1": "POST /process-user-data",
            "2": "POST /process-abap-data"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process-user-data")
async def process_user_data(request: JsonDataRequest):
    try:
        df = processor.process(request.data)
        records = df.to_dict("records")
        summary = dict(df["ambiguity_flag"].value_counts())
        return JSONResponse({
            "status": "success",
            "source": "user_json",
            "ambiguity_summary": summary,
            "processed_data": records,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
            **({"metadata": request.metadata} if request.metadata else {})
        })
    except Exception as e:
        logger.error(f"User data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-abap-data")
async def process_abap_data(request: AbapDataRequest):
    try:
        df = processor.process(request.data)
        records = df.to_dict("records")
        summary = dict(df["ambiguity_flag"].value_counts())
        resp = {
            "status": "success",
            "source": "abap_system",
            "ambiguity_summary": summary,
            "processed_data": records,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
        }
        if request.source_info:
            resp["source_info"] = request.source_info
        if request.table_info:
            resp["table_info"] = request.table_info
        return JSONResponse(resp)
    except Exception as e:
        logger.error(f"ABAP data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For Vercel / local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
