from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dataset to JSON API", description="API for processing datasets in serverless environment")

# Sample data for demonstration (replace with your actual data source)
SAMPLE_DATA = [
    {"id": 1, "name": "Product A", "date": "2025-08-20", "amount": 150.00, "status": "active"},
    {"id": 2, "name": "Product B", "date": "2025-08-19", "amount": 200.50, "status": "inactive"},
    {"id": 3, "name": "Product C", "date": "2025-08-18", "amount": 99.99, "status": "active"},
    {"id": 4, "name": "Product D", "date": "2025-08-17", "amount": 300.25, "status": "active"}
]

class DataSourceRequest(BaseModel):
    source_type: str
    source_path: str
    payload: Optional[Dict[Any, Any]] = None

class DataProcessor:
    """Serverless-compatible data processor"""
    
    def read_sample_data(self) -> pd.DataFrame:
        """Return sample data as DataFrame"""
        df = pd.DataFrame(SAMPLE_DATA)
        logger.info(f"Read {len(df)} sample records")
        return df
    
    def read_from_external_api(self, api_url: str) -> pd.DataFrame:
        """Read data from external API (replace with your actual data source)"""
        try:
            # This is where you'd call your actual data source
            # For now, returning sample data
            df = pd.DataFrame(SAMPLE_DATA)
            logger.info(f"Read {len(df)} records from external source")
            return df
        except Exception as e:
            logger.error(f"Error reading from external source: {str(e)}")
            raise HTTPException(status_code=500, detail=f"External source error: {str(e)}")

# Initialize processor
processor = DataProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Dataset to JSON API is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "serverless",
        "message": "API is running on Vercel"
    }

@app.post("/trigger-csv-from-source")
async def trigger_data_from_specific_source(request: DataSourceRequest):
    """
    Get data from specific source and return as JSON
    """
    try:
        logger.info(f"Data retrieval triggered for {request.source_type}: {request.source_path}")
        
        df = None
        
        if request.source_type.lower() == "sample":
            df = processor.read_sample_data()
        elif request.source_type.lower() == "external_api":
            df = processor.read_from_external_api(request.source_path)
        else:
            # For serverless, we can't use file-based storage
            # You'll need to integrate with cloud storage or external APIs
            raise HTTPException(
                status_code=400, 
                detail="Invalid source_type. Use 'sample' or 'external_api'. File-based sources not supported in serverless."
            )
        
        # Convert DataFrame to JSON-serializable format
        data_dict = df.to_dict('records')
        
        response_data = {
            "status": "success",
            "message": f"Data from {request.source_type} retrieved successfully",
            "source_type": request.source_type,
            "source_path": request.source_path,
            "records_count": len(df),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "data": data_dict,
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data retrieved successfully: {len(df)} records from {request.source_type}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in data retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data retrieval error: {str(e)}")

@app.get("/get-sample-data")
async def get_sample_data():
    """Get sample data directly"""
    try:
        df = processor.read_sample_data()
        data_dict = df.to_dict('records')
        
        return {
            "status": "success",
            "records_count": len(df),
            "data": data_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
