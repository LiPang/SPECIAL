import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import random
import warnings
import torch
import numpy as np
from scipy.io import *
from utils.utility import rsshow
from hyperseg.HyperSeg import hyperseg_classification
import spectral

# Set environment variables and warnings
warnings.filterwarnings('ignore')

def seed_everywhere(seed=42):
    """Set the random seed for reproducibility across all devices."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_paviac():
    label_list = [
        'Background',
        'Water',
        'Trees',
        'Meadows',
        'Self-Blocking Bricks',
        'Bare Soil',
        'Asphalt',
        'Bitumen',
        'Tile',
        'Shadows'
    ]
    data_name = 'PaviaC'
    img_ori = loadmat("data/PaviaCentre/Pavia.mat")['pavia']
    label_gt = loadmat("data/PaviaCentre/Pavia_gt.mat")['pavia_gt']
    img_rgb = loadmat("data/PaviaCentre/data_mat_rgb.mat")['data']

    positions = [(0, 1096, 0, 715)]

    label_copy = label_gt.copy()
    label_list = [label_list[int(t)] for t in np.unique(label_gt)]
    for t, tt in enumerate(np.unique(label_gt)):
        if label_gt.min() > 0:
            label_copy[label_gt == tt] = t + 1  # no background
        else:
            label_copy[label_gt == tt] = t
    if label_gt.min() == 0:
        label_list.pop(0)
    label_gt = label_copy
    return data_name, img_ori, label_gt, label_list, positions, img_rgb



def load_aerorit():
    label_list = ["Background", "Buildings", "River or Lake", "Cars",
                  "Grass or Trees", "Soil or Road"]
    data_name = 'AeroRIT'
    data = spectral.open_image(
        "data/AeroRIT/Reflectance/2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr")
    img_ori_ = np.array(data[:, :, :])
    img_ori = img_ori_[..., 10:-10]

    label_gt = loadmat("data/AeroRIT/label_mat.mat")['data']
    mapping = np.array([0, 1, 4, 5, 2, 3])
    label_gt = mapping[label_gt]
    img_rgb = loadmat("data/AeroRIT/data_mat_rgb.mat")['data']

    img_ori = img_ori[500:500+1024, 100:100+1024*3]
    label_gt = label_gt[500:500+1024, 100:100+1024*3]
    img_rgb = img_rgb[500:500+1024, 100:100+1024*3]
    positions = []
    gap = 512
    for x in range(0, 1024, gap):
        for y in range(0, 1024 * 3, gap):
            positions.append((x, x + gap, y, y + gap))


    label_copy = label_gt.copy()
    label_list = [label_list[int(t)] for t in np.unique(label_gt)]
    for t, tt in enumerate(np.unique(label_gt)):
        if label_gt.min() > 0:
            label_copy[label_gt == tt] = t + 1  # no background
        else:
            label_copy[label_gt == tt] = t
    if label_gt.min() == 0:
        label_list.pop(0)
    label_gt = label_copy
    return data_name, img_ori, label_gt, label_list, positions, img_rgb


def build_settings():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--iterations', type=int, default=20, help='learning iterations')
    parser.add_argument('--warmup_epoches', type=int, default=10)
    parser.add_argument('--epoches', type=int, default=20, help='epoch number')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--data_name', type=str, default='AeroRIT')
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_settings()

    seed_everywhere(42)

    if args.data_name == 'PaviaC':
        data_name, image_hsi, label_gt, label_list, positions, image = load_paviac()
        args.data_name = data_name
    elif args.data_name == 'AeroRIT':
        data_name, image_hsi, label_gt, label_list, positions, image = load_aerorit()
        args.data_name = data_name
        

    image_hsi = rsshow(image_hsi, 0.0001)
    image = rsshow(image, 0.001)

    hyperseg_classification(image, image_hsi, label_list, label_gt, positions, args)
    print('ok')