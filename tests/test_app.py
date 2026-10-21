import cv2,numpy as np,pytest
from fastapi.testclient import TestClient
from app import main
from app.recognition import normalize_plate,valid_indian_plate
def test_plate_normalization():
    assert normalize_plate("ka 03 ab-1234")=="KA03AB1234"
    assert valid_indian_plate("KA03AB1234")
    assert valid_indian_plate("22BH1234AA")
    assert not valid_indian_plate("HELLO")
def test_health_history(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"DB",tmp_path/"test.db")
    client=TestClient(main.app)
    assert client.get("/api/health").status_code==200
    assert client.get("/api/history").json()["total"]==0
def test_image_api(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"DB",tmp_path/"test.db")
    monkeypatch.setattr(main,"recognize",lambda image:[{"plate":"KA03AB1234","box":[5,5,90,20],"detection_confidence":.9,"ocr_confidence":.95,"valid_indian_format":True}])
    ok,buf=cv2.imencode(".png",np.zeros((100,200,3),dtype=np.uint8))
    assert ok
    client=TestClient(main.app)
    response=client.post("/api/recognize/image",files={"file":("test.png",buf.tobytes(),"image/png")})
    assert response.status_code==200
    assert response.json()["results"][0]["plate"]=="KA03AB1234"
    assert client.get("/api/history").json()["total"]==1
def test_invalid_upload():
    client=TestClient(main.app)
    assert client.post("/api/recognize/image",files={"file":("bad.png",b"nonsense","image/png")}).status_code==422
    assert client.post("/api/recognize/image",files={"file":("bad.txt",b"bad","text/plain")}).status_code==415

def test_detect_regions_fallback_without_rcnn_weights(monkeypatch):
    from app import recognition
    monkeypatch.setattr(recognition,"RCNN_PATH","")
    frame=np.zeros((120,240,3),dtype=np.uint8)
    assert isinstance(recognition.detect_regions(frame),list)

def test_rcnn_output_is_filtered_and_converted(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    from app import recognition
    weights=tmp_path/"plate.pt";weights.write_bytes(b"x")
    class FakeModel:
        def __call__(self,images):
            assert images[0].shape==(3,100,200) and images[0].dtype==torch.float32
            return [{
                "boxes":torch.tensor([[10.,20.,70.,40.],[0.,0.,5.,5.],[50.,50.,90.,80.]]),
                "labels":torch.tensor([1,1,2]),
                "scores":torch.tensor([0.9,0.99,0.95]),
            }]
    monkeypatch.setattr(recognition,"RCNN_PATH",str(weights))
    monkeypatch.setattr(recognition,"_rcnn",FakeModel())
    frame=np.zeros((100,200,3),dtype=np.uint8)
    boxes=recognition.detect_regions(frame)
    # label 2 is rejected; remaining boxes are sorted by score, highest first
    assert boxes[0][:4]==(0,0,5,5)
    assert boxes[1][:4]==(10,20,60,20)
    assert len(boxes)==2
