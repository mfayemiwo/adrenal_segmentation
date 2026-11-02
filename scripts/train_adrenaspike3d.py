#!/usr/bin/env python3
"""Train AdrenaSpike3D ROI refiner with the same recipe for spiking/analog."""
import argparse, csv, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.models.adrenaspike3d import AdrenaSpike3D

class Tiles(Dataset):
    def __init__(self,folder,limit=0):
        self.files=sorted(folder.glob("*.npz"))
        if limit:self.files=self.files[:limit]
        if not self.files:raise FileNotFoundError(folder)
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with np.load(self.files[i]) as f:
            return tuple(torch.from_numpy(f[k].astype(np.float32)[None].copy())
                         for k in ("ct","prob","mask"))

def loss_fn(logits,y):
    bce=F.binary_cross_entropy_with_logits(logits,y,pos_weight=logits.new_tensor(4.))
    p=logits.sigmoid()
    dice=1-((2*(p*y).sum((2,3,4))+1)/
            (p.sum((2,3,4))+y.sum((2,3,4))+1)).mean()
    return bce+dice

@torch.no_grad()
def val_score(model,loader,threshold):
    model.eval()
    dices=[]
    for batch in loader:
        ct,p,y=[x.cuda() for x in batch]
        pr=model(ct,p,operating_threshold=threshold).sigmoid()>=0.5
        gt=y>0
        numerator=2*(pr&gt).sum((2,3,4)).float()
        denom=pr.sum((2,3,4))+gt.sum((2,3,4))
        dices.extend(torch.where(denom>0,numerator/denom.clamp_min(1),1.).flatten().cpu().tolist())
    return float(np.mean(dices))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tiles-dir",type=Path,default=Path("runs/adrenaspike3d_tiles"))
    ap.add_argument("--volume-dir",type=Path,default=Path("runs/adrenaspike3d_cache"))
    ap.add_argument("--variant",choices=["snn","analog"],required=True)
    ap.add_argument("--run-name",required=True)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--max-epochs",type=int,default=30)
    ap.add_argument("--patience",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--max-train-tiles",type=int,default=0)
    ap.add_argument("--max-val-tiles",type=int,default=0)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--event-threshold",type=float,default=0.002)
    ap.add_argument("--smoke-test",action="store_true")
    ap.add_argument("--allow-in-sample-backbone",action="store_true",
        help="Explicitly acknowledge single-fitted Stage B probabilities; exploratory feasibility only.")
    a=ap.parse_args()
    if not torch.cuda.is_available():raise SystemExit("MI300X GPU needed via sbatch")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.smoke_test:
        a.max_train_tiles=a.max_val_tiles=4;a.max_epochs=2
    manifest=json.loads((a.volume_dir/"manifest.json").read_text())
    if not manifest.get("out_of_fold_training_probabilities", False):
        if not a.allow_in_sample_backbone:
            raise SystemExit(
                "REFINER TRAINING STOPPED: baseline probabilities on training patients "
                "are in-sample. Supply out-of-fold predictions, or specify "
                "--allow-in-sample-backbone for EXPLORATORY research only."
            )
        print("WARNING: exploratory pilot uses in-sample backbone predictions; "
              "not publication-grade performance evidence.",flush=True)
    threshold=float(manifest["operating_threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid operating threshold in manifest")
    tiles=Tiles(a.tiles_dir/"train")
    patients=sorted({f.stem.rsplit("_g",1)[0] for f in tiles.files})
    if len(patients)<2:
        raise SystemExit("At least two patients are required for disjoint internal training/validation")
    random.Random(42).shuffle(patients)
    hold=set(patients[:max(1,round(0.2*len(patients)))])
    tr_idx=[i for i,f in enumerate(tiles.files) if f.stem.rsplit("_g",1)[0] not in hold]
    va_idx=[i for i,f in enumerate(tiles.files) if f.stem.rsplit("_g",1)[0] in hold]
    if a.max_train_tiles:tr_idx=tr_idx[:a.max_train_tiles]
    if a.max_val_tiles:va_idx=va_idx[:a.max_val_tiles]
    train=DataLoader(torch.utils.data.Subset(tiles,tr_idx),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(torch.utils.data.Subset(tiles,va_idx),batch_size=a.batch_size)
    print(f"Patient-disjoint internal dev: {len(tr_idx)} train / {len(va_idx)} val ROI tiles",flush=True)
    model=AdrenaSpike3D(a.variant,event_threshold=a.event_threshold).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    output=Path("runs")/a.run_name;output.mkdir(parents=True,exist_ok=True)
    best=-1.;stale=0;rows=[]
    for epoch in range(1,a.max_epochs+1):
        model.train();losses=[]
        for batch in train:
            ct,p,y=[v.cuda() for v in batch]
            optim.zero_grad(set_to_none=True)
            loss=loss_fn(model(ct,p,operating_threshold=threshold),y)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            losses.append(float(loss.detach()))
        score=val_score(model,val,threshold)
        improved=score>best+1e-5
        if improved:
            best=score;stale=0
            torch.save(dict(model_state_dict=model.state_dict(),variant=a.variant,
                event_threshold=a.event_threshold,operating_threshold=threshold,
                seed=a.seed,val_roi_dice=score,epoch=epoch,
                out_of_fold_training_probabilities=bool(
                    manifest.get("out_of_fold_training_probabilities",False))),output/"best_model.pt")
        else:stale+=1
        rows.append(dict(epoch=epoch,train_loss=np.mean(losses),val_roi_dice=score,is_best=int(improved)))
        with (output/"metrics.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        print(f"epoch {epoch} train={np.mean(losses):.4f} roi_dice={score:.4f} best={best:.4f}",flush=True)
        if stale>=a.patience:break

if __name__=="__main__":main()
