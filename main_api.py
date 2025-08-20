from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ABAP Data Processing API", description="API for ABAP integration with dual functionality")

# Request models for different scenarios
class ProcessDataRequest(BaseModel):
    data: List[Dict[Any, Any]]
    source_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class GetDataRequest(BaseModel):
    data_type: str
    filters: Optional[Dict[str, Any]] = None
    max_records: Optional[int] = 1000

class DataProcessor:
    """Data processor for ABAP integration"""
    
    def process_incoming_data(self, data: List[Dict[Any, Any]]) -> pd.DataFrame:
        """Process data received from external source (JSON format)"""
        try:
            if not data:
                raise ValueError("No data provided")
            
            df = pd.DataFrame(data)
            logger.info(f"Processed {len(df)} records from incoming data")
            return df
        except Exception as e:
            logger.error(f"Error processing incoming data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data processing error: {str(e)}")
    
    def get_data_for_abap(self, data_type: str, filters: Optional[Dict] = None, max_records: int = 1000) -> pd.DataFrame:
        """Get data to send to ABAP - replace with your actual data source"""
        try:
            # TODO: Replace this with your actual data source
            # This could be:
            # - Database query
            # - External API call
            # - File system read
            # - Environment variables
            # - Cloud storage
            
            # For now, check if data is available via environment variable
            env_data = os.getenv(f'{data_type.upper()}_DATA')
            if env_data:
                try:
                    data = json.loads(env_data)
                    df = pd.DataFrame(data)
                    if max_records:
                        df = df.head(max_records)
                    logger.info(f"Retrieved {len(df)} records of type {data_type} from environment")
                    return df
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in environment variable {data_type.upper()}_DATA")
            
            # If no environment data, return empty DataFrame with message
            logger.warning(f"No data source configured for type: {data_type}")
            return pd.DataFrame({
                "message": [f"No data source configured for {data_type}"],
                "instruction": ["Configure data source via environment variables or external API"],
                "timestamp": [datetime.now().isoformat()]
            })
            
        except Exception as e:
            logger.error(f"Error retrieving data for ABAP: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data retrieval error: {str(e)}")

# Initialize processor
processor = DataProcessor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ABAP Data Processing API is running",
        "functionalities": [
            "POST /process-data - Process JSON data sent to API",
            "POST /get-data-for-abap - Get data for ABAP consumption"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "serverless",
        "configured_data_sources": list(filter(lambda x: x.endswith('_DATA'), os.environ.keys()))
    }

@app.post("/process-data")
async def process_data(request: ProcessDataRequest):
    """
    FUNCTIONALITY 1: Process JSON data sent to the API
    Use this when you have data in JSON format to process
    """
    try:
        logger.info(f"Processing {len(request.data)} records from external source")
        
        # Process the provided data
        df = processor.process_incoming_data(request.data)
        
        # Convert back to JSON format
        processed_data = df.to_dict('records')
        
        # Perform any additional processing here
        # For example: validation, transformation, calculations, etc.
        
        response_data = {
            "status": "success",
            "operation": "data_processing",
            "message": "Data processed successfully",
            "input_records": len(request.data),
            "output_records": len(processed_data),
            "columns": list(df.columns),
            "processed_data": processed_data,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Include source info if provided
        if request.source_info:
            response_data["source_info"] = request.source_info
            
        # Include metadata if provided
        if request.metadata:
            response_data["metadata"] = request.metadata
        
        logger.info(f"Successfully processed {len(processed_data)} records")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/get-data-for-abap")
async def get_data_for_abap(request: GetDataRequest):
    """
    FUNCTIONALITY 2: Provide data to ABAP
    Use this when ABAP needs to retrieve data from the API
    """
    try:
        logger.info(f"ABAP requesting data of type: {request.data_type}")
        
        # Get data for ABAP
        df = processor.get_data_for_abap(
            data_type=request.data_type,
            filters=request.filters,
            max_records=request.max_records
        )
        
        # Convert to JSON format for ABAP consumption
        data_for_abap = df.to_dict('records')
        
        response_data = {
            "status": "success",
            "operation": "data_retrieval",
            "message": f"Data of type '{request.data_type}' retrieved for ABAP",
            "data_type": request.data_type,
            "records_count": len(data_for_abap),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "data": data_for_abap,
            "retrieval_timestamp": datetime.now().isoformat()
        }
        
        # Include applied filters if any
        if request.filters:
            response_data["applied_filters"] = request.filters
            
        # Include record limit info
        if request.max_records:
            response_data["max_records_limit"] = request.max_records
        
        logger.info(f"Successfully provided {len(data_for_abap)} records to ABAP")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving data for ABAP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data retrieval error: {str(e)}")

@app.get("/available-data-types")
async def get_available_data_types():
    """Get list of available data types for ABAP"""
    configured_types = []
    
    # Check which data types are configured via environment variables
    for key in os.environ.keys():
        if key.endswith('_DATA'):
            data_type = key.replace('_DATA', '').lower()
            configured_types.append(data_type)
    
    return {
        "configured_data_types": configured_types,
        "configuration_info": "Set environment variables like 'MATERIALS_DATA', 'ORDERS_DATA' with JSON data",
        "usage": {
            "process_data": "POST /process-data with JSON data in request body",
            "get_data_for_abap": "POST /get-data-for-abap with data_type specification"
        },
        "timestamp": datetime.now().isoformat()
    }

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
