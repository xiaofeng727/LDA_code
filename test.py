import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from argparse import ArgumentParser

from utils import load_Phi, compute_initialization_matrix, psnr, ssim
from utils import imread_CS_py, img2col_py, col2im_CS_py
from LDA_model import LDA

parser = ArgumentParser(description='Learnable Optimization Algorithms (LOA) - Test')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 30, 40, 50}')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--test_dir', type=str, default='data/Set11', help='test dataset directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained model directory')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--epoch', type=int, default=100, help='which epoch model to load')
parser.add_argument('--phase', type=int, default=15, help='which phase model to load')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cs_ratio = args.cs_ratio
layer_num = args.layer_num
block_size = 33

# Load Phi and Qinit
if os.path.exists('Q_init/Q_init_%d.npy' % cs_ratio):
    Qinit = np.load('Q_init/Q_init_%d.npy' % cs_ratio)
else:
    Qinit = compute_initialization_matrix(cs_ratio)

Phi = torch.from_numpy(load_Phi(cs_ratio)).type(torch.FloatTensor).to(device)
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

# Initialize model
model = LDA(layer_num, args.phase)
model = nn.DataParallel(model)
model.to(device)

group_num = 1
learning_rate = 1e-4

# Construct model path
model_path = "./%s/LDA_layer_%d_group_%d_ratio_%d_lr_%.4f/net_params_epoch%d_phase%d.pkl" % \
             (args.model_dir, layer_num, group_num, cs_ratio, learning_rate, args.epoch, args.phase)

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully from", model_path)
else:
    print(
        f"Warning: Model not found at {model_path}. You might be evaluating with a randomly initialized model. Please verify your arguments.")

model.eval()

# Supported image extensions
valid_ext = ('.tif', '.png', '.jpg', '.jpeg', '.bmp')
test_files = [f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_ext)]
psnr_list, ssim_list = [], []

print(f"Testing on {len(test_files)} images from {args.test_dir}...")

with torch.no_grad():
    for filename in test_files:
        file_path = os.path.join(args.test_dir, filename)

        # Read image, convert to grayscale
        img = Image.open(file_path).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Extract blocks according to CS setting
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(img_np)
        img_col = img2col_py(Ipad, block_size)

        batch_x = torch.from_numpy(img_col.T).float().to(device)

        # CS Measurement Operation
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        # Reconstruct by feeding to the LDA Model
        x_output = model(Phix, Phi, Qinit)

        # Reshape the output tensor back into image form
        out_col = x_output.cpu().numpy().T
        rec_img = col2im_CS_py(out_col, row, col, row_new, col_new)

        # clip boundary to [0, 1] for proper visual and metric evaluation
        rec_img = np.clip(rec_img, 0, 1)

        # Convert to proper shapes for metric calculation [Batch, Channel, Height, Width]
        rec_tensor = torch.from_numpy(rec_img).float().unsqueeze(0).unsqueeze(0)
        org_tensor = torch.from_numpy(Iorg).float().unsqueeze(0).unsqueeze(0)

        cur_psnr = psnr(rec_tensor, org_tensor)

        # for ssim from utils, input must be BxCxHxW
        cur_ssim = ssim(rec_tensor, org_tensor).item()

        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)

        print(f"{filename} - PSNR: {cur_psnr:.2f} dB, SSIM: {cur_ssim:.4f}")

if len(psnr_list) > 0:
    print("-" * 40)
    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print("-" * 40)

# # 测试 cs_ratio=25 的模型 (默认参数会在 ./model/ 目录下寻找对应模型)
# python test_model.py --cs_ratio 25

# # 你也可以通过指定参数来测试其他压缩率或其他Epoch的模型
# python test_model.py --cs_ratio 10 --test_dir data/BrainImages_test --epoch 100 --phase 15