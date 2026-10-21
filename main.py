import io,os,sqlite3,time,tempfile,uuid
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,File,UploadFile,HTTPException,Query,Form
from fastapi.responses import FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from .recognition import recognize,valid_indian_plate
BASE=Path(__file__).resolve().parent.parent
DB=Path(os.getenv("PLATE_DB",str(BASE/"platevision.db")))
STATE_CODES={
"AP":"Andhra Pradesh","AR":"Arunachal Pradesh","AS":"Assam","BR":"Bihar","CG":"Chhattisgarh","CH":"Chandigarh","DD":"Dadra and Nagar Haveli and Daman and Diu","DL":"Delhi","GA":"Goa","GJ":"Gujarat","HP":"Himachal Pradesh","HR":"Haryana","JH":"Jharkhand","JK":"Jammu and Kashmir","KA":"Karnataka","KL":"Kerala","LA":"Ladakh","LD":"Lakshadweep","MH":"Maharashtra","ML":"Meghalaya","MN":"Manipur","MP":"Madhya Pradesh","MZ":"Mizoram","NL":"Nagaland","OD":"Odisha","PB":"Punjab","PY":"Puducherry","RJ":"Rajasthan","SK":"Sikkim","TN":"Tamil Nadu","TR":"Tripura","TS":"Telangana","UK":"Uttarakhand","UP":"Uttar Pradesh","WB":"West Bengal"}
def plate_details(plate):
    if not plate:return {"state":None,"rto_code":None,"verification_status":"unreadable"}
    if len(plate)>=4 and plate[:2].isdigit() and plate[2:4]=="BH":
        return {"state":"Bharat Series (BH)","rto_code":None,"verification_status":"format_valid_only"}
    state=STATE_CODES.get(plate[:2]) if valid_indian_plate(plate) else None
    return {"state":state,"rto_code":plate[:4] if state and len(plate)>=4 and plate[2:4].isdigit() else None,"verification_status":"format_valid_only" if state else "format_unrecognized"}
app=FastAPI(title="PlateVision AI")
app.mount("/static",StaticFiles(directory=str(BASE/"static")),name="static")
def connection():
    DB.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB);db.row_factory=sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY,created_at TEXT NOT NULL,source TEXT NOT NULL,plate TEXT,confidence REAL,valid_format INTEGER NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS camera_sessions (session_id TEXT NOT NULL,plate TEXT NOT NULL,PRIMARY KEY(session_id,plate))")
    return db
@app.get("/")
def home():return FileResponse(BASE/"static"/"index.html")
@app.get("/favicon.ico",include_in_schema=False)
def favicon():return RedirectResponse("/static/favicon.svg?v=4",status_code=307)
@app.get("/api/health")
def health():return {"status":"ok","model":"plate-recognizer" if os.getenv("PLATERECOGNIZER_API_TOKEN","").strip() else ("faster-rcnn" if os.getenv("PLATE_RCNN_PATH") and Path(os.getenv("PLATE_RCNN_PATH")).is_file() else "opencv-fallback")}
async def load(file):
    data=await file.read(12*1024*1024+1)
    if len(data)>12*1024*1024:raise HTTPException(413,"File exceeds 12 MB")
    return data
def process(frame,source,session_id=None,debug=False):
    start=time.perf_counter()
    if debug:
        items,trace=recognize(frame,debug=True)
    else:
        items=recognize(frame);trace=None
    for item in items:
        item.update(plate_details(item["plate"]))
        item["plate_validation"]="format_valid" if item["valid_indian_format"] and item["verification_status"]=="format_valid_only" else "format_invalid_or_unknown"
        item["registration_verified"]=False
    with connection() as db:
        for row in items:
            if not row["plate"]: continue
            if source=="camera" and session_id:
                if not row["valid_indian_format"]:continue
                inserted=db.execute("INSERT OR IGNORE INTO camera_sessions(session_id,plate) VALUES(?,?)",(session_id,row["plate"]))
                if inserted.rowcount==0:continue
            elif source=="camera":
                previous=db.execute("SELECT created_at FROM scans WHERE plate=? AND source=? ORDER BY id DESC LIMIT 1",(row["plate"],source)).fetchone()
                if previous and (datetime.now(timezone.utc)-datetime.fromisoformat(previous["created_at"])).total_seconds()<30:continue
            db.execute("INSERT INTO scans(created_at,source,plate,confidence,valid_format) VALUES(?,?,?,?,?)",(datetime.now(timezone.utc).isoformat(),source,row["plate"],row["ocr_confidence"],int(row["valid_indian_format"])))
    response={"results":items,"count":len(items),"processing_ms":round((time.perf_counter()-start)*1000,1),"width":frame.shape[1],"height":frame.shape[0]}
    if debug:response["debug"]={"model":"plate-recognizer" if os.getenv("PLATERECOGNIZER_API_TOKEN","").strip() else ("faster-rcnn" if os.getenv("PLATE_RCNN_PATH") and Path(os.getenv("PLATE_RCNN_PATH")).is_file() else "opencv-fallback"),"trace":trace}
    return response
@app.post("/api/recognize/image")
async def image_scan(file:UploadFile=File(...),debug:bool=Query(False)):
    if file.content_type not in ("image/jpeg","image/png","image/webp"):raise HTTPException(415,"Upload JPEG, PNG or WebP")
    raw=await load(file)
    try:
        image=Image.open(io.BytesIO(raw));image.verify()
        image=Image.open(io.BytesIO(raw)).convert("RGB")
        if image.width*image.height>20000000:raise HTTPException(413,"Image too large")
        frame=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    except (OSError,ValueError,UnidentifiedImageError):raise HTTPException(422,"Invalid image")
    return process(frame,"image",debug=debug)
@app.post("/api/recognize/camera")
async def camera_scan(file:UploadFile=File(...),session_id:str|None=Form(None)):
    if file.content_type not in ("image/jpeg","image/png","image/webp"):raise HTTPException(415,"Upload JPEG, PNG or WebP")
    raw=await load(file)
    try:
        image=Image.open(io.BytesIO(raw)).convert("RGB")
        if image.width*image.height>20000000:raise HTTPException(413,"Image too large")
        frame=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    except (OSError,ValueError,UnidentifiedImageError):raise HTTPException(422,"Invalid camera frame")
    if session_id and (len(session_id)>80 or not all(ch.isalnum() or ch in "-_" for ch in session_id)):raise HTTPException(422,"Invalid camera session")
    return process(frame,"camera",session_id=session_id)
@app.post("/api/recognize/video")
async def video_scan(file:UploadFile=File(...),sample_every:float=Query(2,ge=.5,le=10)):
    if file.content_type not in ("video/mp4","video/webm","video/quicktime","application/octet-stream"):raise HTTPException(415,"Unsupported video type")
    raw=await load(file);suffix=Path(file.filename or "").suffix.lower()
    if suffix not in (".mp4",".mov",".webm"):raise HTTPException(415,"Unsupported video extension")
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as temp:temp.write(raw);path=temp.name
    cap=cv2.VideoCapture(path)
    try:
        if not cap.isOpened():raise HTTPException(422,"Could not decode video")
        fps=cap.get(cv2.CAP_PROP_FPS) or 25;frames_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=frames_count/fps if frames_count else 0
        if duration>120:raise HTTPException(413,"Video too long")
        frames=[];idx=0;step=max(1,round(fps*sample_every))
        while len(frames)<60:
            ok,frame=cap.read()
            if not ok:break
            if idx%step==0:frames.append({"second":round(idx/fps,2),**process(frame,"video")})
            idx+=1
        return {"frames":frames,"sampled_frames":len(frames),"duration_seconds":round(duration,2)}
    finally:cap.release();os.unlink(path)
@app.get("/api/history")
def history(limit:int=Query(30,ge=1,le=200)):
    with connection() as db:
        rows=db.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
        total=db.execute("SELECT count(*) FROM scans").fetchone()[0]
    return {"total":total,"items":[dict(row) for row in rows]}

@app.delete("/api/history")
def clear_history():
    with connection() as db:
        deleted=db.execute("DELETE FROM scans").rowcount
    return {"deleted":deleted,"message":"Scan history cleared"}
