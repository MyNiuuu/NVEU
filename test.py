import argparse
import os
import cv2
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch import nn
import yaml
from model_temporal import Generator_noflow as Generator
from distributed import synchronize


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_dataset(dir):
    nir_images = []
    vis_images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    scenes = os.listdir(dir)
    scenes.sort()
    # scenes = scenes[::-1]
    for scene in scenes:
        middle = os.path.join(dir, scene)
        vis_list = os.listdir(os.path.join(middle, 'VIS'))
        vis_list.sort()
        nir_list = os.listdir(os.path.join(middle, 'NIR'))
        nir_list.sort()
        assert len(vis_list) == len(nir_list), f'{middle}'

        point_vis_list = vis_list[1::][:-1]
        point_nir_list = nir_list[1::][:-1]

        for vis in point_vis_list:
            vis_images.append(os.path.join(middle, 'VIS', vis))
        for nir in point_nir_list:
            nir_images.append(os.path.join(middle, 'NIR', nir))
        
        nir_images.sort()
        vis_images.sort()

    return nir_images, vis_images


def get_data(nir_path, vis_path):
    isp_path = vis_path

    nir_name = os.path.basename(nir_path)
    nir_dir = os.path.dirname(nir_path)

    isp_name = os.path.basename(isp_path)
    isp_dir = os.path.dirname(isp_path)

    nirs, viss, isps = [], [], []

    for i in [-1, 0, 1]:
        nirs.append(cv2.imread(os.path.join(nir_dir, str(int(nir_name.split('.')[0]) + i).zfill(6) + '.png'), 0))
        isps.append(cv2.imread(os.path.join(isp_dir, str(int(isp_name.split('.')[0]) + i).zfill(6) + '.png'))[:, :, ::-1])

    for i in range(len(nirs)):
        nirs[i] = torch.tensor(nirs[i] / float(255), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        isps[i] = torch.tensor(isps[i] / float(255), dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

    return nirs[0], nirs[1], nirs[2], isps[0], isps[1], isps[2]


def test(opts, device, nirlist, vislist):
    for i, (nir_path, rgb_path) in enumerate(zip(nirlist, vislist)):
        print(f'{i}/{len(nirlist)}({len(vislist)})')

        nir0, nir1, nir2, isp0, isp1, isp2 = get_data(nir_path, rgb_path)
        nir0, nir1, nir2 = nir0.to(device), nir1.to(device), nir2.to(device)
        isp0, isp1, isp2 = isp0.to(device), isp1.to(device), isp2.to(device)
        
        with torch.no_grad():
            vis_a_hat, _, _ = g(
                nir0, nir1, nir2, isp0, isp1, isp2
            )
        
        b_fake_rgb = vis_a_hat[0]
        b_rgb_path = rgb_path
        save_dir = os.path.dirname(os.path.join(opts.save_to, os.path.relpath(b_rgb_path, opts.testroot)))
        os.makedirs(save_dir, exist_ok=True)
        
        basename = os.path.basename(rgb_path).replace('bin', 'png')
        l_b_fake_rgb = b_fake_rgb
        l_b_fake_rgb = l_b_fake_rgb.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).detach().to('cpu', torch.uint8).numpy()

        # print(os.path.join(save_dir, basename))
        # assert False

        cv2.imwrite(os.path.join(save_dir, basename), l_b_fake_rgb[:, :, ::-1])
        # assert False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--testroot', type=str, required=True, help="outputs path")
    parser.add_argument('--save_to', type=str, required=True, help="outputs path")
    parser.add_argument("--resume", required=True)
    opts = parser.parse_args()

    cudnn.benchmark = True
    setup_seed(521)

    # Load experiment setting
    device = "cuda"

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    g = Generator(opts).to(device).eval()

    ckpt = opts.resume
    state_dict = torch.load(ckpt)
    # Load generators
    g.load_state_dict(state_dict['g'])

    nirlist, vislist = make_dataset(opts.testroot)

    test(opts, device, nirlist, vislist)
