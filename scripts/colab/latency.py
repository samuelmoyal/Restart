# TensorRT FP16 latency (batch 1) on this GPU for each arch x imgsz; arch only, weights irrelevant.
import os, json, time, glob, numpy as np, torch
from ultralytics import YOLO
OUT='/content/drive/MyDrive/capstone_sim2real/paper/latency_L4.json'
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
img=(np.random.rand(720,1280,3)*255).astype(np.uint8)
for arch in ['yolov8n-seg','yolo11n-seg','yolo26n-seg','yolov8s-seg','yolo11s-seg','yolo26s-seg','yolov8m-seg','yolo11m-seg','yolo26m-seg']:
    for sz in (640,960,1280):
        k=f'{arch}@{sz}'
        if k in res: continue
        try:
            os.makedirs('/content/lat',exist_ok=True); os.chdir('/content/lat')
            eng=YOLO(f'{arch}.pt').export(format='engine', half=True, imgsz=sz, batch=1, verbose=False)
            m=YOLO(eng, task='segment')
            for _ in range(20): m.predict(img, imgsz=sz, verbose=False)
            sp=[]
            for _ in range(200):
                r=m.predict(img, imgsz=sz, verbose=False, conf=0.25)[0]; sp.append(r.speed)
            t={s: float(np.mean([x[s] for x in sp])) for s in ('preprocess','inference','postprocess')}
            tot=[sum(x.values()) for x in sp]
            res[k]={**t,'total_mean':float(np.mean(tot)),'total_p95':float(np.percentile(tot,95)),'gpu':torch.cuda.get_device_name()}
            json.dump(res,open(OUT,'w'),indent=1); print(k,res[k],flush=True)
            for f in glob.glob('/content/lat/*.engine')+glob.glob('/content/lat/*.onnx'): os.remove(f)
        except Exception as e: print('FAIL',k,repr(e),flush=True)
print('LAT DONE')
