# Build /content/rwy: BARS (own split) + RLD (block split), runway class only, YOLO-seg format.
import os, re, zipfile, shutil, hashlib, glob
from concurrent.futures import ThreadPoolExecutor
D='/content/drive/MyDrive'; O='/content/rwy'
for s in ('train','val','test'):
    os.makedirs(f'{O}/images/{s}',exist_ok=True); os.makedirs(f'{O}/labels/{s}',exist_ok=True)
def keep0(t): return ''.join(l+'\n' for l in t.splitlines() if l.split() and l.split()[0]=='0')
# BARS
def cp(a):
    s,f=a; st=os.path.splitext(f)[0]
    shutil.copyfile(f'{D}/BARS_yolo/images/{s}/{f}', f'{O}/images/{s}/bars_{f}')
    lp=f'{D}/BARS_yolo/labels/{s}/{st}.txt'
    open(f'{O}/labels/{s}/bars_{st}.txt','w').write(keep0(open(lp).read()) if os.path.exists(lp) else '')
jobs=[(s,f) for s in ('train','val','test') for f in os.listdir(f'{D}/BARS_yolo/images/{s}') if f.lower().endswith(('.jpg','.png','.jpeg'))]
with ThreadPoolExecutor(32) as ex: list(ex.map(cp,jobs))
print('BARS',len(jobs),flush=True)
# RLD: blocks of 200 consecutive ids -> 80/10/10 by hash; ids like '11510(1)' are duplicates
z=zipfile.ZipFile(f'{D}/Runway Landing Dataset.zip'); n=0
for name in sorted(x for x in z.namelist() if x.endswith('.jpg')):
    st=os.path.basename(name)[:-4].replace('(','_').replace(')',''); b=int(re.match(r'\d+',st).group())//200
    h=int(hashlib.md5(str(b).encode()).hexdigest(),16)%10
    s='test' if h==0 else 'val' if h==1 else 'train'
    open(f'{O}/images/{s}/rld_{st}.jpg','wb').write(z.read(name))
    ln=name.replace('images_rename','labels_rename')[:-4]+'.txt'
    open(f'{O}/labels/{s}/rld_{st}.txt','w').write(keep0(z.read(ln).decode()) if ln in z.NameToInfo else ''); n+=1
print('RLD',n)
for s in ('train','val','test'):
    fs=os.listdir(f'{O}/images/{s}'); print(s,len(fs),sum(f.startswith('bars') for f in fs),sum(f.startswith('rld') for f in fs))
open(f'{O}/rwy.yaml','w').write(f"path: {O}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: runway\n")
# per-source test yamls
for src in ('bars','rld'):
    os.makedirs(f'{O}/test_{src}/images',exist_ok=True); os.makedirs(f'{O}/test_{src}/labels',exist_ok=True)
    for f in glob.glob(f'{O}/images/test/{src}_*'):
        b=os.path.basename(f); st=b.rsplit('.',1)[0]
        if not os.path.lexists(f'{O}/test_{src}/images/{b}'): os.symlink(f,f'{O}/test_{src}/images/{b}')
        if not os.path.lexists(f'{O}/test_{src}/labels/{st}.txt'): os.symlink(f'{O}/labels/test/{st}.txt',f'{O}/test_{src}/labels/{st}.txt')
    open(f'{O}/test_{src}.yaml','w').write(f"path: {O}/test_{src}\ntrain: images\nval: images\nnames:\n  0: runway\n")
print('done')
