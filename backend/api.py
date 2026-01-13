import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import uuid
from datetime import datetime
import json

from backend.modules.run_pipeline import run_full_pipeline


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


app = FastAPI(
    title="Data Quality SaaS API",
    description="Production-grade data quality analysis platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


job_storage = {}


@app.get("/")
def root():
    return {
        "service": "Data Quality Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "clean": "/clean",
            "download": "/download/{job_id}"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(job_storage)
    }


@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    duplicate_strategy: str = "flag"
):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum 100MB allowed")
        
        report, cleaned_df = run_full_pipeline(
            file_bytes=content,
            duplicate_strategy=duplicate_strategy
        )
        
        report = convert_to_serializable(report)
        
        job_id = str(uuid.uuid4())
        
        job_storage[job_id] = {
            "cleaned_df": cleaned_df,
            "timestamp": datetime.now().isoformat(),
            "original_filename": file.filename
        }
        
        return {
            "status": "success",
            "job_id": job_id,
            "report": report,
            "message": "Analysis complete. Use job_id to download cleaned data."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/clean")
async def clean_file(
    file: UploadFile = File(...),
    duplicate_strategy: str = "flag",
    missing_strategy: str = "median",
    outlier_method: str = "clip"
):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum 100MB allowed")
        
        report, cleaned_df = run_full_pipeline(
            file_bytes=content,
            duplicate_strategy=duplicate_strategy
        )
        
        report = convert_to_serializable(report)
        
        job_id = str(uuid.uuid4())
        
        job_storage[job_id] = {
            "cleaned_df": cleaned_df,
            "timestamp": datetime.now().isoformat(),
            "original_filename": file.filename
        }
        
        return {
            "status": "success",
            "job_id": job_id,
            "report": report,
            "summary": {
                "rows_before": report["summary"]["rows_original"],
                "rows_after": report["summary"]["rows_cleaned"],
                "quality_score": report["quality_score"]["score"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")


@app.get("/download/{job_id}")
async def download_cleaned_file(job_id: str):
    
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    try:
        job_data = job_storage[job_id]
        cleaned_df = job_data["cleaned_df"]
        original_filename = job_data["original_filename"]
        
        buffer = BytesIO()
        cleaned_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        filename = original_filename.replace('.csv', '_cleaned.csv')
        
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except Exception as e:
        print(f"Download error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    
    if job_id in job_storage:
        del job_storage[job_id]
        return {"status": "success", "message": "Job deleted"}
    
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs")
async def list_jobs():
    
    jobs = []
    for job_id, data in job_storage.items():
        jobs.append({
            "job_id": job_id,
            "timestamp": data["timestamp"],
            "filename": data["original_filename"],
            "rows": len(data["cleaned_df"])
        })
    
    return {"jobs": jobs, "total": len(jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)