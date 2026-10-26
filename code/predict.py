"""Load a trusted local checkpoint. Supply already-preprocessed historical windows."""
import numpy as np
import torch
from model import BPNet

class Predictor:
    def __init__(self, checkpoint):
        self.ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = BPNet(self.ckpt['scg_channels'], self.ckpt['bcg_channels'])
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.eval()

    def predict(self, scg, bcg, fs):
        if float(fs) != self.ckpt['fs']:
            raise ValueError('Sampling rate differs from training.')
        tensors = []
        for name, signal in [('scg', scg), ('bcg', bcg)]:
            x = np.asarray(signal, dtype=np.float32)
            if x.shape != (self.ckpt[name+'_channels'], self.ckpt['samples']):
                raise ValueError(f'Invalid {name} shape.')
            if not np.isfinite(x).all():
                raise ValueError('Invalid signal values.')
            tensors.append((torch.from_numpy(x)[None]-self.ckpt[name+'_mean']) /
                           self.ckpt[name+'_std'])
        with torch.no_grad():
            sbp, dbp = self.model(*tensors)[0].tolist()
        return {'SBP': sbp, 'DBP': dbp}
