import os,re,threading,cv2
import numpy as np
from pathlib import Path
# Faster R-CNN plate detector. Point PLATE_RCNN_PATH at a state_dict produced by
# tools/train_rcnn.py (2 classes: background, licence plate).
RCNN_PATH=os.getenv("PLATE_RCNN_PATH","")
RCNN_CONF=float(os.getenv("PLATE_RCNN_CONF","0.35"))
_rcnn=None
_rcnn_lock=threading.Lock()
def build_rcnn(num_classes=2):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    # No pretrained download at inference time: weights come from PLATE_RCNN_PATH.
    return fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=num_classes)
def load_rcnn(path):
    import torch
    model=build_rcnn()
    model.load_state_dict(torch.load(path,map_location="cpu"))
    model.eval()
    return model
def detect_rcnn(frame):
    global _rcnn
    import torch
    with _rcnn_lock:
        if _rcnn is None:
            _rcnn=load_rcnn(RCNN_PATH)
    # OpenCV is BGR; torchvision expects RGB float tensors in [0,1], CHW.
    rgb=np.ascontiguousarray(frame[:,:,::-1])
    tensor=torch.from_numpy(rgb).permute(2,0,1).float().div(255.0)
    with torch.inference_mode():
        pred=_rcnn([tensor])[0]
    out=[]
    for box,label,score in zip(pred["boxes"].tolist(),pred["labels"].tolist(),pred["scores"].tolist()):
        if label!=1 or score<RCNN_CONF:continue
        x1,y1,x2,y2=map(int,box)
        out.append((x1,y1,x2-x1,y2-y1,float(score)))
    return sorted(out,key=lambda p:p[4],reverse=True)
def normalize_plate(value):
    return re.sub("[^A-Z0-9]","",value.upper())[:16]
def valid_indian_plate(value):
    return bool(re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",value) or re.fullmatch(r"\d{2}BH\d{4}[A-Z]{1,2}",value))
def detect_regions(frame):
    if RCNN_PATH and Path(RCNN_PATH).is_file():
        return detect_rcnn(frame)
    # Contours are only OCR proposals, NOT confirmed plate detections.
    # Prefer bright, wide, character-dense rectangles; reject grille patterns.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    blur=cv2.bilateralFilter(gray,7,60,60)
    edges=cv2.Canny(blur,65,180)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:350]:
        x,y,bw,bh=cv2.boundingRect(contour)
        ratio=bw/max(bh,1)
        area=bw*bh/(w*h)
        if not (2.0<=ratio<=6.8 and .0015<=area<=.16 and bw>=65 and bh>=14):
            continue
        patch=gray[y:y+bh,x:x+bw]
        if patch.size==0:continue
        # Genuine Indian white plates tend to be brighter than dark grilles.
        bright=float((patch>150).mean())
        if bright<.30:continue
        inside=cv2.Canny(patch,75,170)
        density=float((inside>0).mean())
        if not (.025<=density<=.35):continue
        score=bright*.6+density*.4
        if not any(max(0,min(x+bw,a+c)-max(x,a))*max(0,min(y+bh,b+d)-max(y,b)) > .65*min(bw*bh,c*d) for a,b,c,d,_ in proposals):
            proposals.append((x,y,bw,bh,score))
    return sorted(proposals,key=lambda p:p[4],reverse=True)[:12]
def ocr(crop):
    import pytesseract
    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd=os.environ["TESSERACT_CMD"]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    height,width=gray.shape
    if min(height,width)<8:return "",0
    scale=max(2,min(5,80/max(height,1)))
    gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    variants=[gray,cv2.equalizeHist(gray),cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)]
    candidates=[]
    for img in variants[:2]:
        blurred=cv2.GaussianBlur(img,(3,3),0)
        variants2=[blurred,cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)]
        for prepared in variants2[:2]:
            prepared=cv2.copyMakeBorder(prepared,16,16,16,16,cv2.BORDER_CONSTANT,value=255)
            for psm in (7,13):
                try:
                    data=pytesseract.image_to_data(prepared,config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",output_type=pytesseract.Output.DICT)
                except pytesseract.TesseractNotFoundError:return "",0
                value=normalize_plate("".join(data["text"]))
                scores=[float(c) for c in data["conf"] if float(c)>=0]
                if len(value)>=5:
                    confidence=round(sum(scores)/len(scores)/100,3) if scores else 0
                    candidates.append((value,confidence))
    if not candidates:return "",0
    return max(candidates,key=lambda x:(valid_indian_plate(x[0]),x[1],len(x[0])))
def recognize_cloud(frame):
    """Use the licensed Plate Recognizer API when configured; never expose token to browser."""
    import httpx
    token=os.getenv("PLATERECOGNIZER_API_TOKEN","").strip()
    if not token:return None
    ok,encoded=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise RuntimeError("Unable to encode recognition image")
    try:
        with httpx.Client(timeout=20.0) as client:
            response=client.post(
                "https://api.platerecognizer.com/v1/plate-reader/",
                headers={"Authorization":f"Token {token}"},
                files={"upload":("frame.jpg",encoded.tobytes(),"image/jpeg")},
                data={"regions":"in","mmc":"true"},
            )
            response.raise_for_status()
            payload=response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Plate Recognizer API returned HTTP {exc.response.status_code}") from exc
    except (httpx.RequestError,ValueError) as exc:
        raise RuntimeError("Plate Recognizer API is unavailable") from exc
    results=[]
    for entry in payload.get("results",[]):
        box=entry.get("box") or {}
        x,y=int(box.get("xmin",0)),int(box.get("ymin",0))
        w=max(0,int(box.get("xmax",0))-x)
        h=max(0,int(box.get("ymax",0))-y)
        plate=normalize_plate(entry.get("plate",""))
        if not plate or not w or not h:continue
        vehicle=entry.get("vehicle") or {}
        make_model=entry.get("model_make") or entry.get("make_model") or []
        if isinstance(make_model,list):make_model=make_model[0] if make_model else {}
        if not isinstance(make_model,dict):make_model={}
        vehicle_type=vehicle.get("type") or vehicle.get("value") if isinstance(vehicle,dict) else None
        color=entry.get("color") or []
        if isinstance(color,list):color=color[0] if color else {}
        if not isinstance(color,dict):color={}
        results.append({"plate":plate,"box":[x,y,w,h],
            "vehicle_type":vehicle_type,
            "vehicle_make":make_model.get("make") if str(make_model.get("make","")).lower() not in ("generic","unknown") else None,
            "vehicle_model":make_model.get("model") if str(make_model.get("model","")).lower() not in ("generic","unknown") else None,
            "vehicle_color":color.get("color") or color.get("value"),
            "detection_confidence":float(entry.get("dscore") or 0),
            "ocr_confidence":float(entry.get("score") or 0),
            "valid_indian_format":valid_indian_plate(plate),
            "engine":"plate-recognizer"})
    return results

def recognize(frame):
    if os.getenv('PLATERECOGNIZER_API_TOKEN','').strip():
        return recognize_cloud(frame)
    height,width=frame.shape[:2];items=[]
    for x,y,w,h,score in detect_regions(frame)[:12]:
        x,y=max(x,0),max(y,0);w,h=min(w,width-x),min(h,height-y)
        if w<1 or h<1:continue
        padx=max(3,int(w*.08));pady=max(3,int(h*.18))
        x0=max(0,x-padx);y0=max(0,y-pady)
        x1=min(width,x+w+padx);y1=min(height,y+h+pady)
        plate,confidence=ocr(frame[y0:y1,x0:x1])
        if not valid_indian_plate(plate) or confidence<.30:continue
        items.append({"plate":plate or None,"box":[x,y,w,h],"detection_confidence":score or None,"ocr_confidence":confidence,"valid_indian_format":valid_indian_plate(plate)})
    return sorted(items,key=lambda item:(item['valid_indian_format'],item['ocr_confidence']),reverse=True)
