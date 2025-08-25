from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel
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

    def __init__(self, z_threshold=1.5):
        # Threshold for z-score to be considered an anomaly
        self.z_threshold = z_threshold

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only numeric columns for detection"""
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            raise ValueError("No numeric columns available for anomaly detection")
        return df_numeric.fillna(0)

    def detect_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run z-score anomaly detection on given dataframe"""
        df_num = self.clean_numeric_data(df)
        anomalies = np.zeros(len(df_num), dtype=bool)

        # Compute z-score for each numeric column, flag if any column is anomalous
        for col in df_num.columns:
            values = df_num[col].values
            mean = np.mean(values)
            std = np.std(values)
            # Avoid divide by zero
            if std == 0:
                z_scores = np.zeros_like(values)
            else:
                z_scores = (values - mean) / std
            col_anomalies = np.abs(z_scores) > self.z_threshold
            anomalies = anomalies | col_anomalies

        df["fraud_prediction"] = np.where(anomalies, "fraud", "legit")
        logger.info("Z-score anomaly detection completed")
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
processor = DataProcessor(z_threshold=1.5)  # 3 standard deviations from mean

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API is running (using Z-score)",
        "endpoints": {
            "1": "POST /process-user-data - Process & detect fraud (anomalies) in JSON data",
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
        fraud_summary = {str(k): int(v) for k, v in df["fraud_prediction"].value_counts().items()}
        return {
            "status": "success",
            "source": "user_json",
            "fraud_summary": fraud_summary,
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-abap-data")
async def process_abap_data_raw(data: List[Dict[Any, Any]] = Body(...)):
    try:
        df = processor.process_abap_data(data)
        processed_data = df.to_dict('records')
        fraud_summary = {str(k): int(v) for k, v in df["fraud_prediction"].value_counts().items()}
        return {
            "status": "success",
            "source": "abap_system",
            "fraud_summary": fraud_summary,
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
