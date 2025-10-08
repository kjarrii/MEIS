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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.layer import MultiStepContainer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


DATASET_DIR = ['easyvoxelsx/']
MODEL_SAVE_DIR = '31models/'
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
    'weight_decay':  0.0,
    'scheduler':     'StepLR',
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('snn_spatial')
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
        self.height = 360
        self.width = 640

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


class ResidualSNNBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        ch  = cfg['channels']
        tau = cfg['tau']

        self.conv1 = MultiStepContainer(nn.Conv2d(in_channels, ch, kernel_size=3, padding=1, bias=False)        )
        self.bn1   = MultiStepContainer(nn.BatchNorm2d(ch))
        self.lif1  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv2 = MultiStepContainer(nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)        )
        self.bn2   = MultiStepContainer(nn.BatchNorm2d(ch))

        self.skip_conv = MultiStepContainer(nn.Conv2d(in_channels, ch, kernel_size=1, bias=False))
        self.skip_bn   = MultiStepContainer(nn.BatchNorm2d(ch))
        self.lif2  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.skip_conv(identity)
        identity = self.skip_bn(identity)

        out = out + identity
        out = self.lif2(out)
        return out


class ResidualBinaryNet(nn.Module):
    def __init__(self, in_channels, height, width, cfg):
        super().__init__()
        ch  = cfg['channels']
        tau = cfg['tau']

        self.block1 = ResidualSNNBlock(in_channels, cfg)
        self.pool1  = MultiStepContainer(nn.MaxPool2d(2))

        self.block2 = ResidualSNNBlock(ch, cfg)
        self.pool2  = MultiStepContainer(nn.MaxPool2d(2))

        self.block3 = ResidualSNNBlock(ch, cfg)
        self.pool3  = MultiStepContainer(nn.MaxPool2d(2))

        Hf = height // (2**3)
        Wf = width  // (2**3)
        D  = ch * Hf * Wf

        self.fc       = MultiStepContainer(nn.Linear(D, 2, bias=True))
        self.lif_out  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)

        self.apply(self._init_weights)
        self.spikes = []

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        self.spikes = []
        x = x.permute(1, 0, 2, 3, 4)

        x = self.block1(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool1(x)

        x = self.block2(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool2(x)

        x = self.block3(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        x = self.lif_out(x)
        logits = x.sum(dim=0)
        return logits


class Trainer:
    def __init__(self, cfg, logger):
        self.config = cfg
        self.logger = logger
        self.device = DEVICE
        self._prepare()

    def _prepare(self):
        dataset = H5VoxelDataset(DATASET_DIR)
        labels  = [1 if int(os.path.splitext(os.path.basename(p))[0])<10000 else 0 for p in dataset.files]
        pos = [i for i,l in enumerate(labels) if l==1]
        neg = [i for i,l in enumerate(labels) if l==0]
        random.shuffle(pos); random.shuffle(neg)
        tp = max(1, int(0.176*len(pos)))
        tn = max(1, int(0.176*len(neg)))
        val_idx   = pos[:tp] + neg[:tn]
        train_idx = pos[tp:] + neg[tn:]

        bs = self.config['batch_size']
        train_labels = [labels[i] for i in train_idx]
        w_pos = 1.0/sum(train_labels)
        w_neg = 1.0/(len(train_labels)-sum(train_labels))
        weights = [w_pos if l==1 else w_neg for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=False)

        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=bs, sampler=sampler)
        self.val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=bs, shuffle=False)

        self.model = ResidualBinaryNet(dataset.channels, dataset.height, dataset.width, self.config).to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        if self.config['scheduler']=='StepLR':
            self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.1)
        else:
            self.sched = None

        self.criterion = nn.CrossEntropyLoss()
        self.scaler    = torch.amp.GradScaler() if USE_AMP else None

    def train_one_epoch(self, epoch):
        self.model.train()
        all_preds, all_labels = [], []
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x,y in pbar:
            x,y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = self.model(x)
                loss = self.criterion(out, y)
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
            pbar.set_postfix({'loss': running_loss/len(all_labels), 'spikes:': self.model.spikes})

        if self.sched: self.sched.step()
        loss_avg = running_loss / len(all_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        return loss_avg, {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

    def validate(self, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x,y in tqdm(self.val_loader, desc=f"Eval {epoch}"):
                x,y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                preds = out.argmax(dim=1)
                all_preds += preds.cpu().tolist()
                all_labels += y.cpu().tolist()
                val_loss += loss.item()*y.size(0)
                functional.reset_net(self.model)
        loss_avg = val_loss / len(all_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn)
        return loss_avg, {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'acc':acc}

    def run(self):
        best_acc = 0.0
        for ep in range(1, EPOCHS+1):
            #try:
            tr_loss, tr_cm = self.train_one_epoch(ep)
            val_loss, val_cm = self.validate(ep)
            log = {'epoch':ep,'train_loss':tr_loss,'train_cm':tr_cm,'val_loss':val_loss,'val_cm':val_cm}
            self.logger.info(json.dumps(log))
            if val_cm['acc'] > best_acc:
                best_acc = val_cm['acc']
                torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best.pth'))
        #except Exception as e:
            tb = traceback.format_exc()
            #self.logger.info(json.dumps({'error':str(e),'traceback':tb,'epoch':ep}))
        torch.save(self.model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'final.pth'))


if __name__=='__main__':
    set_seed()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger = setup_logger(LOG_FILE)
    Trainer(config, logger).run()
    print(f"Logs at {LOG_FILE}")
