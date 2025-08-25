from fastapi import FastAPI, HTTPException
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel
import numpy as np
from sklearn.metrics import accuracy_score

# Configure logging for debugging and info purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with title and description for auto API docs
app = FastAPI(title="Fraud Detection API", description="API to process data and detect fraud from ABAP or user")

# Define Pydantic request model for generic JSON data input
class JsonDataRequest(BaseModel):
    data: List[Dict[Any, Any]]          # List of transaction or record dictionaries
    metadata: Optional[Dict[str, Any]] = None  # Optional extra metadata describing the data

# Define Pydantic request model for ABAP specific data input
class AbapDataRequest(BaseModel):
    data: List[Dict[Any, Any]]          # Data table records from ABAP
    source_info: Optional[Dict[str, Any]] = None  # Optional SAP source metadata
    table_info: Optional[Dict[str, Any]] = None   # Optional SAP table information

# Main class handling the fraud detection logic
class DataProcessor:
    
    def __init__(self, z_threshold=1.5):
        """
        Initialize with z-score threshold for anomaly detection.
        Any numeric feature with absolute z-score > z_threshold marks the record as fraud.
        """
        self.z_threshold = z_threshold

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only numeric columns from input dataframe for anomaly detection.
        Raises error if no numeric columns found.
        """
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            raise ValueError("No numeric columns available for anomaly detection")
        # Fill any missing numeric values with zero for safe calculation
        return df_numeric.fillna(0)

    def detect_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform z-score anomaly detection on numeric columns.
        Marks rows as 'fraud' if any numeric value is beyond threshold in z-score.
        Returns original dataframe with an added 'fraud_prediction' column.
        """
        df_num = self.clean_numeric_data(df)
        # Boolean array to keep track of anomalies across all numeric columns
        anomalies = np.zeros(len(df_num), dtype=bool)

        for col in df_num.columns:
            values = df_num[col].values
            mean = np.mean(values)
            std = np.std(values)
            # Avoid division by zero when std=0 (all values are same)
            if std == 0:
                z_scores = np.zeros_like(values)
            else:
                z_scores = (values - mean) / std
            # Flag records exceeding threshold as anomalies
            col_anomalies = np.abs(z_scores) > self.z_threshold
            # Combine anomaly flags from each column
            anomalies = anomalies | col_anomalies

        # Add fraud prediction column based on combined anomalies
        df["fraud_prediction"] = np.where(anomalies, "fraud", "legit")
        logger.info("Z-score anomaly detection completed")
        return df

    def calculate_accuracy(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate accuracy of the prediction if true labels ('true_label' column) exist.
        Converts fraud/legit strings to binary 1/0 for accuracy_score computation.
        Returns accuracy as decimal or None if no true labels present.
        """
        if "true_label" in df.columns:
            y_true = df["true_label"].values
            y_pred = df["fraud_prediction"].values
            # Map string labels to binary
            y_true_bin = np.array([1 if str(x).lower() == "fraud" else 0 for x in y_true])
            y_pred_bin = np.array([1 if str(x).lower() == "fraud" else 0 for x in y_pred])
            return accuracy_score(y_true_bin, y_pred_bin)
        return None

    def process_user_json_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """
        Entry point for processing generic user-provided JSON data.
        Converts the list of dicts to DataFrame and applies fraud detection.
        """
        if not data:
            raise ValueError("No data provided")
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} user records")
        return self.detect_fraud(df)

    def process_abap_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """
        Entry point for processing ABAP system data.
        Converts the list of dicts to DataFrame and applies fraud detection.
        """
        if not data:
            raise ValueError("No data provided from ABAP")
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} ABAP records")
        return self.detect_fraud(df)

# Instantiate processor with default z-score threshold
processor = DataProcessor(z_threshold=1.5)

# Root endpoint, gives metadata about the API
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

# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Endpoint to accept user JSON data and detect fraud
@app.post("/process-user-data")
async def process_user_data(request: JsonDataRequest):
    try:
        # Detect fraud predictions
        df = processor.process_user_json_data(request.data)
        # Calculate accuracy if possible
        accuracy = processor.calculate_accuracy(df)
        # Prepare response data
        processed_data = df.to_dict('records')
        fraud_summary = {str(k): int(v) for k, v in df["fraud_prediction"].value_counts().items()}
        response = {
            "status": "success",
            "source": "user_json",
            "fraud_summary": fraud_summary,
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }
        # Add accuracy as percentage or N/A if no true labels
        response["accuracy"] = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to accept ABAP data and detect fraud
@app.post("/process-abap-data")
async def process_abap_data(request: AbapDataRequest):
    try:
        df = processor.process_abap_data(request.data)
        accuracy = processor.calculate_accuracy(df)
        processed_data = df.to_dict('records')
        fraud_summary = {str(k): int(v) for k, v in df["fraud_prediction"].value_counts().items()}
        response = {
            "status": "success",
            "source": "abap_system",
            "fraud_summary": fraud_summary,
            "processed_data": processed_data,
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
            "abap_source_info": request.source_info,
            "table_info": request.table_info
        }
        response["accuracy"] = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
        return response
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run app with uvicorn on port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
