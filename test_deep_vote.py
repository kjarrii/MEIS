import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from spikingjelly.clock_driven.layer import MultiStepContainer
from spikingjelly.clock_driven import neuron, surrogate, functional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


TALA = 21
MODEL_PATH = '../' +  str(TALA) + 'models/best.pth'
DATA_DIRS   = ['easyvoxelsx/', 'medvoxelsx/', 'hardvoxelsx/']
NUM_BINS   = 10

config = {
    'batch_size':    16,
    'learning_rate': 1e-2,
    'tau':           5.0,
    'dropout':       0.2,
    'channels':      8,
    'weight_decay':  0.0,
    'scheduler':     'StepLR',
    'readout_gain':  50.0
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


class MySNNModel(nn.Module):
    def __init__(self, in_channels, height, width, cfg):
        super().__init__()
        ch  = cfg['channels']
        tau = cfg['tau']

        self.conv1 = MultiStepContainer(nn.Conv2d(in_channels, ch, 3, padding=1, bias=False))
        self.bn1   = MultiStepContainer(nn.BatchNorm2d(ch))
        self.lif1  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pool1 = MultiStepContainer(nn.MaxPool2d(2))

        self.conv2 = MultiStepContainer(nn.Conv2d(ch, ch, 3, padding=1, bias=False))
        self.bn2   = MultiStepContainer(nn.BatchNorm2d(ch))
        self.lif2  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pool2 = MultiStepContainer(nn.MaxPool2d(2))

        self.conv3 = MultiStepContainer(nn.Conv2d(ch, ch, 3, padding=1, bias=False))
        self.bn3   = MultiStepContainer(nn.BatchNorm2d(ch))
        self.lif3  = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pool3 = MultiStepContainer(nn.MaxPool2d(2))

        Hf = height // (2**3)
        Wf = width  // (2**3)
        D  = ch * Hf * Wf

        self.fc     = MultiStepContainer(nn.Linear(D, 2, bias=True))
        self.lif_out = neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.readout_gain = nn.Parameter(torch.tensor(cfg['readout_gain']))
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
        self.spikes = []
        x = x.permute(1, 0, 2, 3, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lif3(x)
        self.spikes.append(int(x.sum().item()))
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        x = self.lif_out(x)
        x = x * self.readout_gain
        return x.sum(dim=0)


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
            model = MySNNModel(2,360, 640, config).to(device)
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
