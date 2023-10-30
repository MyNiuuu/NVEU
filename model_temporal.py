import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from package_core.net_basics import conv2d, Cascade_resnet_blocks
from detectron2.layers import ModulatedDeformConv
from network_swinir import SwinIR_NIR_Encoder


def actFunc(act, *args, **kwargs):
	act = act.lower()
	if act == 'relu':
		return nn.ReLU()
	elif act == 'relu6':
		return nn.ReLU6()
	elif act == 'leakyrelu':
		return nn.LeakyReLU(0.1)
	elif act == 'prelu':
		return nn.PReLU()
	elif act == 'rrelu':
		return nn.RReLU(0.1, 0.3)
	elif act == 'selu':
		return nn.SELU()
	elif act == 'celu':
		return nn.CELU()
	elif act == 'elu':
		return nn.ELU()
	elif act == 'gelu':
		return nn.GELU()
	elif act == 'tanh':
		return nn.Tanh()
	else:
		raise NotImplementedError


def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ImageEncoder(nn.Module):
	def __init__(self, in_chs, init_chs):
		super(ImageEncoder, self).__init__()
		self.conv0 = conv2d(
			in_planes=in_chs,
			out_planes=init_chs,
			batch_norm=False,
			activation=nn.ReLU(),
			kernel_size=7,
			stride=1
		)
		self.resblocks0 = Cascade_resnet_blocks(in_planes=init_chs, n_blocks=1)  # green block in paper
		self.conv1 = conv2d(
			in_planes=init_chs,
			out_planes=2 * init_chs,
			batch_norm=False,
			activation=nn.ReLU(),
			kernel_size=3,
			stride=2
		)
		self.resblocks1 = Cascade_resnet_blocks(in_planes=2 * init_chs, n_blocks=1)
		self.conv2 = conv2d(
			in_planes=2 * init_chs,
			out_planes=2 * init_chs,
			batch_norm=False,
			activation=nn.ReLU(),
			kernel_size=3,
			stride=2
		)
		self.resblocks2 = Cascade_resnet_blocks(in_planes=2 * init_chs, n_blocks=1)

	def forward(self, x):
		x0 = self.resblocks0(self.conv0(x))
		x1 = self.resblocks1(self.conv1(x0))
		x2 = self.resblocks2(self.conv2(x1))
		return x2, x1, x0


class ModulatedDeformLayer(nn.Module):
	"""
	Modulated Deformable Convolution (v2)
	"""

	def __init__(self, in_chs, out_chs, kernel_size=3, deformable_groups=1, activation='relu'):
		super(ModulatedDeformLayer, self).__init__()
		assert isinstance(kernel_size, (int, list, tuple))
		self.deform_offset = conv3x3(in_chs, (3 * kernel_size ** 2) * deformable_groups)
		self.act = actFunc(activation)
		self.deform = ModulatedDeformConv(
			in_chs,
			out_chs,
			kernel_size,
			stride=1,
			padding=1,
			deformable_groups=deformable_groups
		)

	def forward(self, x, feat):
		offset_mask = self.deform_offset(feat)
		offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
		offset = torch.cat((offset_x, offset_y), dim=1)
		mask = mask.sigmoid()
		out = self.deform(x, offset, mask)
		out = self.act(out)
		return out


class Generator_noflow(nn.Module):
	def __init__(self, opts, in_chs=3, out_chs=3, n_feats=64, embed_dim=48):
		super(Generator_noflow, self).__init__()
		self.swin = SwinIR_NIR_Encoder(
			img_size=(240, 320), window_size=8, img_range=1., 
			depths=[6, 6], embed_dim=embed_dim, 
			num_heads=[6, 6], mlp_ratio=2
		)
		
		self.deform_0 = ModulatedDeformLayer(3 * embed_dim, n_feats, deformable_groups=8)
		self.resblocks_0 = Cascade_resnet_blocks(n_feats, n_blocks=3)
		self.deform_1 = ModulatedDeformLayer(n_feats, n_feats, deformable_groups=8)
		self.resblocks_1 = Cascade_resnet_blocks(n_feats, n_blocks=3)
		self.to_RGB = nn.Conv2d(n_feats, out_chs, kernel_size=3, stride=1, padding=1)
	
	def forward(self, nir_0, nir_1, nir_2, isp_0, isp_1, isp_2):
		fea0 = self.swin(nir_0, isp_0)
		fea1 = self.swin(nir_1, isp_1)
		fea2 = self.swin(nir_2, isp_2)

		x = torch.cat([fea0, fea1, fea2], dim=1)
		x = self.deform_0(x, x)
		x = self.resblocks_0(x)
		x = self.deform_1(x, x)
		x = self.resblocks_1(x)
		x = self.to_RGB(x)

		return x, [x, x, x, x], [x, x, x, x]

