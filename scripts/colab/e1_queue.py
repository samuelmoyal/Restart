# Sequential, resumable training queue. One line per run in QUEUE: name model imgsz epochs [extra k=v ...]
import os, sys, json, time, shutil, subprocess
from ultralytics import YOLO
OUT='/content/drive/MyDrive/capstone_sim2real/paper/e1'; os.makedirs(OUT, exist_ok=True)
QUEUE=f'{OUT}/queue.txt'
def runs():
    return [l.split() for l in open(QUEUE) if l.strip() and not l.startswith('#')]
done=set()
while True:
    todo=[r for r in runs() if r[0] not in done and not os.path.exists(f'{OUT}/{r[0]}/result.json')]
    if not todo: break
    name, model, imgsz, epochs, *extra = todo[0]; done.add(name)
    kw={k: (float(v) if v.replace('.','',1).isdigit() else v) for k,v in (e.split('=') for e in extra)}
    t0=time.time()
    try:
        m=YOLO(model)
        m.train(data='/content/rwy/rwy.yaml', imgsz=int(imgsz), epochs=int(epochs), batch=kw.pop('batch',16) if 'batch' in kw else 16,
                project='/content/runs', name=name, exist_ok=True, workers=8, seed=0, deterministic=False, plots=False, verbose=False, **kw)
        best=f'/content/runs/{name}/weights/best.pt'
        res={'name':name,'model':model,'imgsz':int(imgsz),'epochs':int(epochs),'train_min':(time.time()-t0)/60,**kw}
        for src in ('bars','rld'):
            v=YOLO(best).val(data=f'/content/rwy/test_{src}.yaml', imgsz=int(imgsz), batch=16, split='val', plots=False, verbose=False)
            res[src]={'mask_map50':v.seg.map50,'mask_map':v.seg.map,'box_map50':v.box.map50,'mask_p':v.seg.mp,'mask_r':v.seg.mr}
        os.makedirs(f'{OUT}/{name}', exist_ok=True)
        shutil.copy(best, f'{OUT}/{name}/best.pt')
        shutil.copy(f'/content/runs/{name}/results.csv', f'{OUT}/{name}/results.csv')
        json.dump(res, open(f'{OUT}/{name}/result.json','w'), indent=1)
        print('DONE', json.dumps(res), flush=True)
    except Exception as e:
        print('FAIL', name, repr(e), flush=True)
print('QUEUE EMPTY', flush=True)
