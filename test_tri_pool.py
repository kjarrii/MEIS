import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from spikingjelly.clock_driven.layer import MultiStepContainer, SeqToANNContainer, MultiStepDropout
from spikingjelly.clock_driven import neuron, surrogate, functional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


TALA = 11
MODEL_PATH = '../' + str(TALA) + 'models/best.pth'
DATA_DIRS   = ['easyvoxelsx/', 'medvoxelsx/', 'hardvoxelsx/']
NUM_BINS   = 10

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

class VoxelDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        self.time_steps = NUM_BINS
        self.channels = 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        num = int(os.path.splitext(os.path.basename(path))[0])
        with h5py.File(path, 'r') as f:
            if 'voxel' in f:
                data_np = f['voxel'][()]
            else:
                data_np = f['data'][()]
        data_n = (data_np - data_np.mean()) / (data_np.std() + 1e-5) * 2.0
        return torch.from_numpy(data_n).float(), num, data_np

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    for DATA_DIR in DATA_DIRS:
        print(DATA_DIR)
        dataset = VoxelDataset(DATA_DIR)
        if len(dataset) == 0:
            print(f"No .h5 files found in {DATA_DIR}. Exiting.")
            return
        counter = tn = fn = fp = tp = 0
        fns = []
        fps = []
        true_labels = []
        predicted_probs = []
        for voxel, num, vox in dataset:
            x = voxel.unsqueeze(0).to(device)
            model = CextBinaryNet(2, config).to(device)
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            functional.reset_net(model)
            with torch.no_grad():
                output = model(x)

            predicted = output.argmax(dim=1)

            probs = output.squeeze(0).cpu().tolist()
            truth = 0 if num > 10000 else 1

            if predicted == 0 and truth == 0:
                tn += 1
            elif predicted == 0 and truth == 1:
                fn += 1
                fns.append(num)
            elif predicted == 1 and truth == 0:
                fp += 1
                fps.append(num)
            elif predicted == 1 and truth == 1:
                tp += 1

            true_labels.append(truth)
            predicted_probs.append(probs[1])
            counter += 1

        print(f"{{'tn': {tn}, 'fn': {fn}, 'fp': {fp}, 'tp': {tp}}}")
        print(f"Total samples: {counter}")
        print('fns:', fns)
        print('fps:', fps)

        fpr, tpr, roc_thresholds = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)

        print("\n# ROC Data")
        print("False Positive Rates (fpr):", fpr.tolist())
        print("True Positive Rates  (tpr):", tpr.tolist())
        print("ROC Thresholds        :", roc_thresholds.tolist())
        print(f"AUC Score             : {roc_auc:.6f}")

        precision, recall, pr_thresholds = precision_recall_curve(true_labels, predicted_probs)
        avg_prec = average_precision_score(true_labels, predicted_probs)

        print("\n# Precision-Recall Data")
        print("Precision values      :", precision.tolist())
        print("Recall values         :", recall.tolist())
        print("PR Thresholds         :", pr_thresholds.tolist())
        print(f"Average Precision (AP): {avg_prec:.6f}")

if __name__ == "__main__":
    main()
