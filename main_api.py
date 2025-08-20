from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse
import pandas as pd
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from pydantic import BaseModel

print("hello")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dataset to CSV API", description="API triggered by ABAP to process complete datasets")

# Configuration - FIXED: Use /tmp directory which is writable on Vercel
CSV_OUTPUT_DIR = Path("/tmp/csv_exports")
CSV_OUTPUT_DIR.mkdir(exist_ok=True)

DATABASE_PATH = "data.db"  # SQLite database path
DATA_FILE_PATH = "dataset.xlsx"  # Excel file path (alternative data source)

# Pydantic models for request bodies
class CSVExportRequest(BaseModel):
    source_type: str
    source_path: str
    payload: Optional[Dict[Any, Any]] = None

class DataProcessor:
    """Class to handle data reading and CSV conversion"""
    
    def __init__(self):
        self.last_export_time = None
        self.ensure_database_exists()  # Add this line
        
    def ensure_database_exists(self):
        """Ensure database file exists, create if not"""
        if not os.path.exists(DATABASE_PATH):
            logger.warning(f"Database file {DATABASE_PATH} not found. Creating empty database.")
            try:
                # Create empty database with a sample table
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                
                # Create a sample table (adjust columns based on your needs)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS main_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        date TEXT,
                        amount REAL,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert sample data (optional)
                sample_data = [
                    ("Sample Record 1", "2025-08-20", 100.50, "active"),
                    ("Sample Record 2", "2025-08-19", 250.75, "inactive"),
                    ("Sample Record 3", "2025-08-18", 175.25, "active")
                ]
                
                cursor.executemany('''
                    INSERT INTO main_data (name, date, amount, status) 
                    VALUES (?, ?, ?, ?)
                ''', sample_data)
                
                conn.commit()
                conn.close()
                logger.info(f"Database {DATABASE_PATH} created successfully with sample data")
                
            except Exception as e:
                logger.error(f"Failed to create database: {str(e)}")
                raise
    
    def read_from_database(self, table_name: str = "main_data") -> pd.DataFrame:
        """Read complete dataset from SQLite database"""
        try:
            if not os.path.exists(DATABASE_PATH):
                raise FileNotFoundError(f"Database file not found: {DATABASE_PATH}")
                
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                conn.close()
                raise ValueError(f"Table '{table_name}' does not exist in database")
            
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Read {len(df)} records from database table: {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading from database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Initialize data processor
processor = DataProcessor()

@app.post("/trigger-csv-export")
async def trigger_csv_export(
    background_tasks: BackgroundTasks,
    payload: Optional[Dict[Any, Any]] = None
):
    """
    Main endpoint triggered by ABAP when new entry is added.
    Reads complete dataset and converts to CSV.
    """
    try:
        logger.info("CSV export triggered by ABAP")

        # Log the payload from ABAP for debugging
        if payload:
            logger.info(f"ABAP payload: {json.dumps(payload, indent=2)}")

        # Read complete dataset (try database first, then Excel, then sample data for testing)
        try:
            df = processor.read_from_database()
            data_source = "database"
        except Exception as db_error:
            logger.warning(f"Database read failed: {db_error}. Trying Excel file...")
            try:
                df = processor.read_from_excel()
                data_source = "excel_file"
            except Exception as excel_error:
                logger.warning(f"Excel read failed: {excel_error}. Using sample data for testing...")
                df = processor.create_sample_data()
                data_source = "sample_data"

        # Generate CSV filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"complete_dataset_{timestamp}.csv"

        # Save to CSV
        csv_path = processor.save_to_csv(df, csv_filename)

        # Process in background to return response quickly to ABAP
        background_tasks.add_task(log_export_completion, csv_path, len(df))

        response_data = {
            "status": "success",
            "message": "Dataset exported to CSV successfully",
            "data_source": data_source,
            "records_count": len(df),
            "columns_count": len(df.columns),
            "csv_filename": csv_filename,
            "csv_path": csv_path,
            "export_timestamp": datetime.now().isoformat(),
            "columns": list(df.columns)
        }

        logger.info(f"Export completed: {csv_filename} with {len(df)} records")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in CSV export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# FIXED: Multiple versions of the endpoint to handle different parameter methods
from pydantic import BaseModel

# Add this request model
class DataSourceRequest(BaseModel):
    source_type: str
    source_path: str
    payload: Optional[Dict[Any, Any]] = None

@app.post("/trigger-csv-from-source")
async def trigger_csv_from_specific_source(
    request: DataSourceRequest
):
    """
    Alternative endpoint to get data from specific data source.
    source_type: 'database', 'excel', or 'csv'
    source_path: path to file or table name for database
    Returns: Complete dataset as JSON
    """
    try:
        logger.info(f"Data retrieval triggered for {request.source_type}: {request.source_path}")
        
        if request.source_type.lower() == "database":
            df = processor.read_from_database(request.source_path)
        elif request.source_type.lower() == "excel":
            df = processor.read_from_excel(request.source_path)
        elif request.source_type.lower() == "csv":
            df = processor.read_from_csv(request.source_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid source_type. Use 'database', 'excel', or 'csv'")
        
        # Convert DataFrame to dictionary/JSON format
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


@app.get("/export-status")
async def get_export_status():
    """Get status of last export operation"""
    try:
        csv_files = list(CSV_OUTPUT_DIR.glob("*.csv"))
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if csv_files:
            latest_file = csv_files[0]
            file_stats = latest_file.stat()

            return {
                "last_export_file": latest_file.name,
                "file_size_bytes": file_stats.st_size,
                "created_timestamp": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_timestamp": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "total_csv_files": len(csv_files)
            }
        else:
            return {
                "message": "No CSV exports found",
                "total_csv_files": 0
            }
    except Exception as e:
        return {
            "error": f"Error accessing export status: {str(e)}",
            "total_csv_files": 0
        }

@app.get("/list-exports")
async def list_all_exports():
    """List all CSV export files"""
    try:
        csv_files = list(CSV_OUTPUT_DIR.glob("*.csv"))

        file_list = []
        for file in csv_files:
            stats = file.stat()
            file_list.append({
                "filename": file.name,
                "size_bytes": stats.st_size,
                "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })

        # Sort by modification time, newest first
        file_list.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "total_files": len(file_list),
            "files": file_list
        }
    except Exception as e:
        return {
            "error": f"Error listing exports: {str(e)}",
            "total_files": 0
        }

@app.delete("/cleanup-exports")
async def cleanup_old_exports(keep_latest: int = Query(5, description="Number of files to keep")):
    """Clean up old CSV export files, keeping only the latest N files"""
    try:
        csv_files = list(CSV_OUTPUT_DIR.glob("*.csv"))
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if len(csv_files) <= keep_latest:
            return {
                "message": f"No cleanup needed. Only {len(csv_files)} files exist.",
                "files_kept": len(csv_files),
                "files_deleted": 0
            }

        files_to_delete = csv_files[keep_latest:]
        deleted_count = 0

        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old export file: {file.name}")
            except Exception as e:
                logger.error(f"Failed to delete {file.name}: {str(e)}")

        return {
            "message": f"Cleanup completed. Kept {keep_latest} latest files.",
            "files_kept": len(csv_files) - deleted_count,
            "files_deleted": deleted_count
        }
    except Exception as e:
        return {
            "error": f"Error during cleanup: {str(e)}",
            "files_kept": 0,
            "files_deleted": 0
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "csv_output_dir": str(CSV_OUTPUT_DIR),
        "tmp_writable": os.access("/tmp", os.W_OK),
        "database_exists": os.path.exists(DATABASE_PATH),
        "excel_file_exists": os.path.exists(DATA_FILE_PATH),
        "endpoints": [
            "/trigger-csv-export",
            "/trigger-csv-from-source (JSON body)",
            "/trigger-csv-from-source-params (query params)",
            "/export-status",
            "/list-exports",
            "/cleanup-exports",
            "/health"
        ]
    }

@app.get("/test-sample")
async def test_with_sample_data():
    """Test endpoint that creates and returns sample data"""
    try:
        df = processor.create_sample_data()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"sample_data_{timestamp}.csv"
        csv_path = processor.save_to_csv(df, csv_filename)

        return {
            "status": "success",
            "message": "Sample data created and saved",
            "records_count": len(df),
            "csv_filename": csv_filename,
            "csv_path": csv_path,
            "data": df.to_dict('records')[:3]  # Show first 3 records
        }
    except Exception as e:
        logger.error(f"Error in sample data test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

async def log_export_completion(csv_path: str, record_count: int):
    """Background task to log export completion"""
    logger.info(f"Background task: Export to {csv_path} completed with {record_count} records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
