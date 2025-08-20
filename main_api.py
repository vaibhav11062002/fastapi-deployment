from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

print("hello")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dataset to CSV API", description="API triggered by ABAP to process complete datasets")

# Configuration
CSV_OUTPUT_DIR = Path("csv_exports")
CSV_OUTPUT_DIR.mkdir(exist_ok=True)

DATABASE_PATH = "data.db"  # SQLite database path
DATA_FILE_PATH = "dataset.xlsx"  # Excel file path (alternative data source)

class DataProcessor:
    """Class to handle data reading and CSV conversion"""
    
    def __init__(self):
        self.last_export_time = None
        
    def read_from_database(self, table_name: str = "main_data") -> pd.DataFrame:
        """Read complete dataset from SQLite database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Read {len(df)} records from database table: {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error reading from database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    def read_from_excel(self, file_path: str = DATA_FILE_PATH) -> pd.DataFrame:
        """Read complete dataset from Excel file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            df = pd.read_excel(file_path)
            logger.info(f"Read {len(df)} records from Excel file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Excel file error: {str(e)}")
    
    def read_from_csv(self, file_path: str) -> pd.DataFrame:
        """Read complete dataset from existing CSV file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Read {len(df)} records from CSV file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"CSV file error: {str(e)}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dataset_export_{timestamp}.csv"
            
            csv_path = CSV_OUTPUT_DIR / filename
            df.to_csv(csv_path, index=False)
            
            self.last_export_time = datetime.now()
            logger.info(f"Saved {len(df)} records to CSV: {csv_path}")
            return str(csv_path)
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            raise HTTPException(status_code=500, detail=f"CSV save error: {str(e)}")

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
        
        # Read complete dataset (try database first, then Excel as fallback)
        try:
            df = processor.read_from_database()
            data_source = "database"
        except Exception as db_error:
            logger.warning(f"Database read failed: {db_error}. Trying Excel file...")
            try:
                df = processor.read_from_excel()
                data_source = "excel_file"
            except Exception as excel_error:
                logger.error(f"Both database and Excel read failed.")
                raise HTTPException(
                    status_code=500, 
                    detail="No data source available. Check database and Excel file."
                )
        
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

@app.post("/trigger-csv-from-source")
async def trigger_csv_from_specific_source(
    source_type: str,
    source_path: str,
    background_tasks: BackgroundTasks,
    payload: Optional[Dict[Any, Any]] = None
):
    """
    Alternative endpoint to export CSV from specific data source.
    source_type: 'database', 'excel', or 'csv'
    source_path: path to file or table name for database
    """
    try:
        logger.info(f"CSV export triggered for {source_type}: {source_path}")
        
        if source_type.lower() == "database":
            df = processor.read_from_database(source_path)
        elif source_type.lower() == "excel":
            df = processor.read_from_excel(source_path)
        elif source_type.lower() == "csv":
            df = processor.read_from_csv(source_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid source_type. Use 'database', 'excel', or 'csv'")
        
        # Generate CSV filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{source_type}_export_{timestamp}.csv"
        
        # Save to CSV
        csv_path = processor.save_to_csv(df, csv_filename)
        
        # Background task
        background_tasks.add_task(log_export_completion, csv_path, len(df))
        
        response_data = {
            "status": "success",
            "message": f"Data from {source_type} exported to CSV successfully",
            "source_type": source_type,
            "source_path": source_path,
            "records_count": len(df),
            "columns_count": len(df.columns),
            "csv_filename": csv_filename,
            "csv_path": csv_path,
            "export_timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in source-specific export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/export-status")
async def get_export_status():
    """Get status of last export operation"""
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

@app.get("/list-exports")
async def list_all_exports():
    """List all CSV export files"""
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

@app.delete("/cleanup-exports")
async def cleanup_old_exports(keep_latest: int = 5):
    """Clean up old CSV export files, keeping only the latest N files"""
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "csv_output_dir": str(CSV_OUTPUT_DIR),
        "database_exists": os.path.exists(DATABASE_PATH),
        "excel_file_exists": os.path.exists(DATA_FILE_PATH)
    }

async def log_export_completion(csv_path: str, record_count: int):
    """Background task to log export completion"""
    logger.info(f"Background task: Export to {csv_path} completed with {record_count} records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
