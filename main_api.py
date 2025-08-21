from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest  # ML for fraud detection
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", description="API to process data and detect fraud from ABAP or user")

# Request models
class JsonDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    metadata: Optional[Dict[str, Any]] = None

class AbapDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    source_info: Optional[Dict[str, Any]] = None
    table_info: Optional[Dict[str, Any]] = None

class DataProcessor:
    """Handles data processing and fraud detection"""

    def __init__(self):
        # Initialize fraud model (can later be retrained dynamically)
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric columns for ML model"""
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            raise ValueError("No numeric columns available for fraud detection")
        return df_numeric.fillna(0)

    def detect_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run anomaly detection on given dataframe"""
        df_num = self.clean_numeric_data(df)

        # Train model on data
        self.model.fit(df_num)

        # Predictions: -1 = anomaly (fraud), 1 = normal
        df["fraud_prediction"] = self.model.predict(df_num)
        df["fraud_prediction"] = df["fraud_prediction"].map({-1: "fraud", 1: "legit"})

        logger.info("Fraud detection completed")
        return df

    def process_user_json_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        if not data:
            raise ValueError("No data provided")
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} user records")
        return self.detect_fraud(df)

    def process_abap_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        if not data:
            raise ValueError("No data provided from ABAP")
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} ABAP records")
        return self.detect_fraud(df)


# Initialize processor
processor = DataProcessor()

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API is running",
        "endpoints": {
            "1": "POST /process-user-data - Process & detect fraud in JSON data",
            "2": "POST /process-abap-data - Detect fraud in ABAP data"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process-user-data")
async def process_user_data(request: JsonDataRequest):
    try:
        df = processor.process_user_json_data(request.data)
        processed_data = df.to_dict('records')
        return {
            "status": "success",
            "source": "user_json",
            "fraud_summary": dict(df["fraud_prediction"].value_counts()),
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-abap-data")
async def process_abap_data(request: AbapDataRequest):
    try:
        df = processor.process_abap_data(request.data)
        processed_data = df.to_dict('records')
        return {
            "status": "success",
            "source": "abap_system",
            "fraud_summary": dict(df["fraud_prediction"].value_counts()),
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
            "abap_source_info": request.source_info,
            "table_info": request.table_info
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# For Vercel / local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
