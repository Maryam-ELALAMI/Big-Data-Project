from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List
import shutil
import os
import uuid
import json
import asyncio
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for demo
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for scan states (for demo purposes)
scans = {}

class ScanOption(str):
    k_gram = "k_gram"
    winnowing = "winnowing"

@app.post("/api/scan")
async def start_scan(files: List[UploadFile] = File(...), options: str = None):
    scan_id = str(uuid.uuid4())
    scan_dir = os.path.join(UPLOAD_DIR, scan_id)
    os.makedirs(scan_dir, exist_ok=True)
    
    file_names = []
    
    # Save uploaded files
    for file in files:
        file_path = os.path.join(scan_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_names.append(file.filename)
    
    # Initialize scan state
    scans[scan_id] = {
        "status": "processing",
        "progress": 0,
        "logs": [],
        "files": file_names,
        "results": None,
        "start_time": datetime.now()
    }
    
    # Start background processing (simulation)
    asyncio.create_task(process_scan(scan_id))
    
    return {"scanId": scan_id, "message": "Scan started"}

async def process_scan(scan_id):
    """Simulate processing steps"""
    scan = scans[scan_id]
    
    steps = [
        ("Initializing analysis engine...", 10),
        ("Parsing source code AST...", 30),
        ("Generating token streams...", 50),
        ("Computing fingerprints...", 70),
        ("Comparing pairs...", 90),
        ("Finalizing report...", 100)
    ]
    
    for log, progress in steps:
        await asyncio.sleep(1) # Simulate work
        scan["logs"].append({"time": datetime.now().strftime("%H:%M:%S"), "message": log})
        scan["progress"] = progress
    
    # Generate mock results based on actual filenames
    # Generate deterministic results based on actual content
    files = scan["files"]
    pairs = []
    
    import random # Keep for simulating processing delay only
    
    def calculate_similarity(text1, text2):
        # fast and simple Jaccard similarity on tokens
        if not text1 or not text2:
            return 0.0
        
        # Simple tokenization: split by whitespace and non-alphanumeric
        def tokenize(text):
            return set(text.lower().split())
            
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        if not tokens1 and not tokens2:
            return 100.0
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return (intersection / union) * 100.0 if union > 0 else 0.0

    # Read all files content first
    file_contents = {}
    for filename in files:
        try:
            file_path = os.path.join(UPLOAD_DIR, scan_id, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents[filename] = f.read()
        except Exception:
            file_contents[filename] = ""

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file_a = files[i]
            file_b = files[j]
            
            similarity = calculate_similarity(file_contents[file_a], file_contents[file_b])
            
            pairs.append({
                "file_a": file_a,
                "file_b": file_b,
                "similarity": round(similarity, 1),
                "label": "high" if similarity > 70 else "medium" if similarity > 40 else "low",
                "overlap_spans": [] # Would contain actual line numbers in real implementation
            })
            
    # Sort pairs by similarity
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
    scan["results"] = {
        "meta": {
            "n_files": len(files),
            "n_pairs": len(pairs),
            "runtime_ms": 3500
        },
        "pairs": pairs
    }
    scan["status"] = "complete"

@app.get("/api/scan/{scan_id}/status")
async def get_scan_status(scan_id: str):
    if scan_id not in scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    scan = scans[scan_id]
    return {
        "status": scan["status"],
        "progress": scan["progress"],
        "logs": scan["logs"],
        "complete": scan["status"] == "complete"
    }

@app.get("/api/scan/{scan_id}/results")
async def get_scan_results(scan_id: str):
    if scan_id not in scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scans[scan_id]["status"] != "complete":
        return {"status": "processing"}
        
    return scans[scan_id]["results"]

@app.get("/api/files/{scan_id}/{filename}")
async def get_file_content(scan_id: str, filename: str):
    """Serve the actual file content for comparison view"""
    file_path = os.path.join(UPLOAD_DIR, scan_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return {"content": content}
    except UnicodeDecodeError:
        try:
             with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
                return {"content": content}
        except:
            return {"content": "Error: Could not decode file content."}

@app.get("/api/samples")
async def get_samples():
    return [
        {"name": "student1.py", "size": 1024, "url": "/samples/student1.py"},
        {"name": "student2.py", "size": 2048, "url": "/samples/student2.py"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
