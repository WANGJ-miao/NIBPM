"""Requires real, synchronized, quality-controlled cuff-paired windows."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import BPNet

def main():
    p = argparse.ArgumentParser()
    p.add_argument('data', help='NPZ: scg, bcg, y, subject_id, fs')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--checkpoint', default='best_bp.pt')
    args = p.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    with np.load(args.data, allow_pickle=False) as d:
        arrays = [d[k].astype(np.float32) for k in ['scg', 'bcg', 'y']]
        ids, fs = d['subject_id'], float(d['fs'].item())
    a, b, y = arrays
    if a.ndim != 3 or b.ndim != 3 or y.shape != (len(a), 2):
        raise ValueError('Expected SCG/BCG [N,C,T], y [N,2].')
    if len(b) != len(a) or ids.shape != (len(a),) or a.shape[2] != b.shape[2]:
        raise ValueError('Inconsistent samples, subject IDs or window lengths.')
    if not np.isfinite(fs) or fs <= 0 or a.shape[2] < 64:
        raise ValueError('Invalid sampling rate or window length.')
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError('Non-finite data; inspect artifacts first.')
    if np.any(y <= 0) or np.any(y[:, 0] <= y[:, 1]):
        raise ValueError('Labels must be positive mmHg, ordered [SBP, DBP].')
    subjects = np.unique(ids)
    if len(subjects) < 10:
        raise ValueError('Starter split requires >=10 subjects; this is not a sample-size recommendation.')
    rng.shuffle(subjects)
    n = max(1, round(len(subjects)*0.15))
    te, va = np.isin(ids, subjects[:n]), np.isin(ids, subjects[n:2*n])
    tr = ~(te | va)
    checkpoint = dict(fs=fs, samples=a.shape[2], scg_channels=a.shape[1],
                      bcg_channels=b.shape[1], label_order=['SBP', 'DBP'])
    for i, name in enumerate(['scg', 'bcg']):
        mean = arrays[i][tr].mean(axis=(0, 2), keepdims=True)
        std = np.maximum(arrays[i][tr].std(axis=(0, 2), keepdims=True), 1e-6)
        arrays[i] = (arrays[i]-mean)/std
        checkpoint[name+'_mean'] = torch.from_numpy(mean)
        checkpoint[name+'_std'] = torch.from_numpy(std)
    for name, mask in [('train', tr), ('val', va), ('test', te)]:
        checkpoint[name+'_subjects'] = np.unique(ids[mask]).astype(str).tolist()

    def loader(mask, shuffle=False):
        return DataLoader(TensorDataset(*[torch.from_numpy(v[mask]) for v in arrays]),
                          batch_size=32, shuffle=shuffle)

    train_dl, val_dl, test_dl = loader(tr, True), loader(va), loader(te)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BPNet(a.shape[1], b.shape[1]).to(device)
    # Initialize from TRAINING labels only.
    with torch.no_grad():
        mean_y = y[tr].mean(axis=0)
        model.head[-1].weight.zero_()
        model.head[-1].bias[0] = float(mean_y[1])
        model.head[-1].bias[1] = float(np.log(np.expm1((mean_y[0]-mean_y[1])/40))-0.54132485)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss(beta=5.0)

    def evaluate(dl):
        model.eval()
        pred, target = [], []
        with torch.no_grad():
            for scg, bcg, bp in dl:
                pred.append(model(scg.to(device), bcg.to(device)).cpu().numpy())
                target.append(bp.numpy())
        return np.concatenate(pred), np.concatenate(target)

    best, stale, state = float('inf'), 0, None
    for epoch in range(args.epochs):
        model.train()
        for scg, bcg, bp in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(scg.to(device), bcg.to(device)), bp.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        pred, target = evaluate(val_dl)
        score = float(np.abs(pred-target).mean())
        print(f'Epoch {epoch+1}: validation MAE {score:.3f} mmHg')
        if score < best:
            best, stale = score, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    if state is None:
        raise RuntimeError('No valid checkpoint; inspect numerical stability.')
    model.load_state_dict(state)
    checkpoint['state_dict'] = state
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    pred, target = evaluate(test_dl)
    error = pred-target
    print('Order [SBP, DBP], units mmHg')
    print('Window MAE:', np.abs(error).mean(axis=0))
    print('Bias:', error.mean(axis=0))
    print('Error SD:', error.std(axis=0, ddof=1))
    print('Train-mean baseline MAE:', np.abs(y[tr].mean(axis=0)-target).mean(axis=0))
    per_subject = [np.abs(error[ids[te] == sid]).mean(axis=0) for sid in np.unique(ids[te])]
    print('Subject-macro MAE:', np.mean(per_subject, axis=0))

if __name__ == '__main__':
    main()
