from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Processing API", description="API to receive and process data from users and ABAP")

# Request models
class JsonDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    metadata: Optional[Dict[str, Any]] = None

class AbapDataRequest(BaseModel):
    data: List[Dict[Any, Any]]  # Data sent from ABAP
    source_info: Optional[Dict[str, Any]] = None  # ABAP system info
    table_info: Optional[Dict[str, Any]] = None   # Table details

class DataProcessor:
    """Data processor for both user and ABAP data"""
    
    def process_user_json_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """Process JSON data sent by users"""
        try:
            if not data:
                raise ValueError("No data provided")
            
            df = pd.DataFrame(data)
            logger.info(f"Processed {len(df)} records from user JSON data")
            return df
        except Exception as e:
            logger.error(f"Error processing user JSON data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"User data processing error: {str(e)}")
    
    def process_abap_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """Process data sent from ABAP system"""
        try:
            if not data:
                raise ValueError("No data provided from ABAP")
            
            df = pd.DataFrame(data)
            logger.info(f"Processed {len(df)} records from ABAP system")
            return df
        except Exception as e:
            logger.error(f"Error processing ABAP data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ABAP data processing error: {str(e)}")

# Initialize processor
processor = DataProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Data Processing API is running",
        "functionalities": {
            "1": "POST /process-user-data - Process JSON data sent by users",
            "2": "POST /process-abap-data - Process data sent from ABAP system"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "serverless"
    }

@app.post("/process-user-data")
async def process_user_data(request: JsonDataRequest):
    """
    FUNCTIONALITY 1: Process JSON data sent by users
    """
    try:
        logger.info(f"Processing user JSON data with {len(request.data)} records")
        
        # Process the JSON data from user
        df = processor.process_user_json_data(request.data)
        
        # Convert to JSON format
        processed_data = df.to_dict('records')
        
        # Add any processing logic here
        # For example: validation, calculations, transformations
        
        response_data = {
            "status": "success",
            "source": "user_json",
            "message": "User JSON data processed successfully",
            "input_records": len(request.data),
            "output_records": len(processed_data),
            "columns": list(df.columns),
            "processed_data": processed_data,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Include metadata if provided
        if request.metadata:
            response_data["metadata"] = request.metadata
        
        logger.info(f"Successfully processed {len(processed_data)} user records")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing user data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User data processing error: {str(e)}")

@app.post("/process-abap-data")
async def process_abap_data(request: AbapDataRequest):
    """
    FUNCTIONALITY 2: Process data sent FROM ABAP system
    """
    try:
        logger.info(f"Processing ABAP data with {len(request.data)} records")
        
        # Process the data sent from ABAP
        df = processor.process_abap_data(request.data)
        
        # Convert to JSON format
        processed_data = df.to_dict('records')
        
        # Add any ABAP-specific processing logic here
        # For example: SAP data validation, field mapping, etc.
        
        response_data = {
            "status": "success",
            "source": "abap_system",
            "message": "ABAP data processed successfully",
            "input_records": len(request.data),
            "output_records": len(processed_data),
            "columns": list(df.columns),
            "processed_data": processed_data,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Include ABAP source info if provided
        if request.source_info:
            response_data["abap_source_info"] = request.source_info
            
        # Include table info if provided
        if request.table_info:
            response_data["table_info"] = request.table_info
        
        logger.info(f"Successfully processed {len(processed_data)} ABAP records")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing ABAP data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ABAP data processing error: {str(e)}")

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
