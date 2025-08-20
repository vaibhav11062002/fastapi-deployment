from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import io

print("hello")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dataset to CSV API", description="API triggered by ABAP to process complete datasets")

# Configuration - Use /tmp directory which is writable on Vercel
CSV_OUTPUT_DIR = Path("/tmp/csv_exports")
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

    def save_to_csv_tmp(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save DataFrame to CSV file in /tmp directory"""
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

    def dataframe_to_csv_stream(self, df: pd.DataFrame) -> io.StringIO:
        """Convert DataFrame to CSV stream without saving to disk"""
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            logger.info(f"Created CSV stream with {len(df)} records")
            return csv_buffer
        except Exception as e:
            logger.error(f"Error creating CSV stream: {str(e)}")
            raise HTTPException(status_code=500, detail=f"CSV stream error: {str(e)}")

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

        # Save to CSV in /tmp directory
        csv_path = processor.save_to_csv_tmp(df, csv_filename)

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
            "columns": list(df.columns),
            "note": "File saved in temporary storage - will be cleared after function execution"
        }

        logger.info(f"Export completed: {csv_filename} with {len(df)} records")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in CSV export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/download-csv")
async def download_csv_direct(
    table_name: str = "main_data"
):
    """
    Alternative endpoint that returns CSV data directly without saving to disk.
    Better for serverless environments.
    """
    try:
        logger.info(f"Direct CSV download requested for table: {table_name}")

        # Read data
        try:
            df = processor.read_from_database(table_name)
            data_source = "database"
        except Exception:
            try:
                df = processor.read_from_excel()
                data_source = "excel"
            except Exception:
                raise HTTPException(status_code=500, detail="No data source available")

        # Create CSV stream
        csv_stream = processor.dataframe_to_csv_stream(df)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_export_{timestamp}.csv"

        # Return CSV as streaming response
        return StreamingResponse(
            io.BytesIO(csv_stream.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct CSV download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.get("/export-status")
async def get_export_status():
    """Get status of last export operation from /tmp directory"""
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
                "total_csv_files": len(csv_files),
                "storage_location": "/tmp (temporary)",
                "note": "Files in /tmp are temporary and cleared between function executions"
            }
        else:
            return {
                "message": "No CSV exports found in temporary storage",
                "total_csv_files": 0
            }
    except Exception as e:
        return {
            "error": f"Cannot access temporary storage: {str(e)}",
            "total_csv_files": 0
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
        "excel_file_exists": os.path.exists(DATA_FILE_PATH)
    }

async def log_export_completion(csv_path: str, record_count: int):
    """Background task to log export completion"""
    logger.info(f"Background task: Export to {csv_path} completed with {record_count} records")

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
