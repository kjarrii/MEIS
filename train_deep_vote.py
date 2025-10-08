import os
import glob
import random
import json
import logging
import traceback

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.layer import SeqToANNContainer, MultiStepDropout
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

DATASET_DIR = ['easyvoxelsx/']
MODEL_SAVE_DIR = '11models/'
LOG_FILE = os.path.join(MODEL_SAVE_DIR, 'results.txt')
EPOCHS = 100
USE_AMP = True
SEED = 23
NUM_BINS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'batch_size':    16,
    'learning_rate': 1e-2,
    'tau':           5.0,
    'dropout':       0.2,
    'channels':      8,
    'hidden_dim':    512,
    'weight_decay':  0.0,
    'scheduler':     'StepLR',
    'use_cupy':      True
}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('snn_trainer')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger


class H5VoxelDataset(Dataset):
    def __init__(self, data_dirs):
        files = []
        for dir in data_dirs:
            for filepath in glob.glob(os.path.join(dir, '*.h5')):
                files.append(filepath)
        self.files = sorted(files)
        self.time_steps = NUM_BINS
        self.channels = 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        num = int(os.path.splitext(os.path.basename(path))[0])
        label = 1 if num < 10000 else 0
        with h5py.File(path, 'r') as f:
            data = f['voxel'][()] if 'voxel' in f else f['data'][()]
        data = (data - data.mean()) / (data.std() + 1e-5) * 2.0
        return torch.from_numpy(data).float(), label


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        return self.voting(x.unsqueeze(1)).squeeze(1)


class CextBinaryNet(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        ch = cfg['channels']
        tau = cfg['tau']
        dropout = cfg['dropout']
        backend = 'torch'

        conv_layers = []
        conv_layers.extend(self.conv3x3(in_channels, ch, tau, backend))
        conv_layers.append(SeqToANNContainer(nn.MaxPool2d(2)))
        for _ in range(4):
            conv_layers.extend(self.conv3x3(ch, ch, tau, backend))
            conv_layers.append(SeqToANNContainer(nn.MaxPool2d(2)))
        self.conv = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Flatten(2),
            MultiStepDropout(dropout),
            SeqToANNContainer(nn.Linear(ch*11*20, cfg['hidden_dim'], bias=True)),
            neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False, backend=backend),
            MultiStepDropout(dropout),
            SeqToANNContainer(nn.Linear(cfg['hidden_dim'], 2, bias=True)),
            neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False, backend=backend, v_threshold=0.5),
            SeqToANNContainer(nn.Softmax(dim=1))
       )

        self.vote = VotingLayer(1)

    @staticmethod
    def conv3x3(in_ch, out_ch, tau, backend):
        return [
            SeqToANNContainer(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)),
            SeqToANNContainer(nn.BatchNorm2d(out_ch)),
            neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False, backend=backend, v_threshold=0.5)
        ]

    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        out_spikes = self.fc(self.conv(x))
        out_rate = out_spikes.mean(0)
        return self.vote(out_rate)


class Trainer:
    def __init__(self, cfg, logger):
        self.config = cfg
        self.logger = logger
        self.device = DEVICE
        self._prepare()

    def _prepare(self):
        dataset = H5VoxelDataset(DATASET_DIR)
        labels = [1 if int(os.path.splitext(os.path.basename(p))[0])<10000 else 0 for p in dataset.files]
        pos = [i for i,l in enumerate(labels) if l==1]
        neg = [i for i,l in enumerate(labels) if l==0]
        random.shuffle(pos); random.shuffle(neg)
        tp = max(1, int(0.176*len(pos)))
        tn = max(1,int(0.176*len(neg)))
        val_idx = pos[:tp] + neg[:tn]
        train_idx = pos[tp:] + neg[tn:]
        bs = self.config['batch_size']

        train_labels = [labels[i] for i in train_idx]
        w_pos = 1.0 / sum(train_labels)
        w_neg = 1.0 / (len(train_labels)-sum(train_labels))
        weights = [w_pos if l==1 else w_neg for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs, sampler=sampler)
        self.val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=bs, shuffle=False)

        self.model = CextBinaryNet(dataset.channels, self.config).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        if self.config['scheduler']=='StepLR':
            self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=5, gamma=0.5)
        else:
            self.sched = None

        self.criterion = nn.CrossEntropyLoss()
        if USE_AMP:
            self.scaler = torch.amp.GradScaler()

    def train_one_epoch(self, epoch):
        self.model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x,y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            y_onehot = F.one_hot(y, 2).float()
            self.opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = self.model(x)
                loss = self.criterion(out, y_onehot)
            if USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward(); self.opt.step()

            preds = out.argmax(dim=1)
            all_preds += preds.cpu().tolist()
            all_labels += y.cpu().tolist()
            running_loss += loss.item()*y.size(0)
            functional.reset_net(self.model)
            pbar.set_postfix({'loss':running_loss/len(all_labels)})
        if self.sched: self.sched.step()
        loss_avg = running_loss/len(all_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        return loss_avg, {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x,y in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                y_onehot = F.one_hot(y,2).float()
                out = self.model(x)
                loss = self.criterion(out, y_onehot)
                preds = out.argmax(dim=1)
                all_preds += preds.cpu().tolist()
                all_labels += y.cpu().tolist()
                val_loss += loss.item()*y.size(0)
                functional.reset_net(self.model)
        loss_avg = val_loss/len(all_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn)
        return loss_avg, {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'acc':acc}

    def run(self):
        best_acc = 0.0
        for ep in range(1, EPOCHS+1):
            try:
                train_loss, train_cm = self.train_one_epoch(ep)
                val_loss, val_cm = self.validate(ep)
                log_entry = {'epoch':ep,'train_loss':train_loss,'train_cm':train_cm,'val_loss':val_loss,'val_cm':val_cm}
                self.logger.info(json.dumps(log_entry))
                if val_cm['acc']>best_acc:
                    best_acc=val_cm['acc']
                    torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"best_lr{self.config['learning_rate']}.pth"))
            except Exception as e:
                tb = traceback.format_exc()
                self.logger.info(json.dumps({'error':str(e),'traceback':tb,'epoch':ep}))
        torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"final_lr{self.config['learning_rate']}.pth"))


if __name__=='__main__':
    set_seed()
    os.makedirs(MODEL_SAVE_DIR,exist_ok=True)
    logger = setup_logger(LOG_FILE)
    trainer = Trainer(config, logger)
    trainer.run()
    print(f"Results logged to {LOG_FILE}")
