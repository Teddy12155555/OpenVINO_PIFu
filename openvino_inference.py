# %%
from torch.nn import init
# from lib.model.HGPIFuNet import HGPIFuNet
# from lib.model.SurfaceClassifier import SurfaceClassifier
# from lib.model.DepthNormalizer import DepthNormalizer
from torchsummary import summary
from torchvision import transforms
from openvino.runtime import (Core)
from PIL import Image
from skimage import measure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import cv2
import os 
import glob
import tqdm

# %%

# %%

import openvino.runtime


# %%
DATAROOT = "./data"
LOADSIZE = 512
NAME = "pifu_ov"
DEBUG = False
NUM_VIEWS = 1
RANDOM_MULTIVIEW = False
GPU_ID = 0
GPU_IDS = 0
NUM_THREADS = 1
SERIAL_BATCHES = False
PIN_MEMORY = False
BATCH_SIZE = 1
LEARNING_RATE = 0.001
LEARNING_RATEC = 0.001
NUM_EPOCH = 100
FREQ_PLOT = 10
FREQ_SAVE = 50
FREQ_SAVE_PLY = 100
NO_GEN_MESH = False
NO_NUM_EVAL = False
RESUME_EPOCH = -1
CONTINUE_TRAIN = False
RESOLUTION = 256
TEST_FOLDER_PATH = "./input_images"
SIGMA = 5.0
NUM_SAMPLE_INOUT = 5000
NUM_SAMPLE_COLOR = 0
Z_SIZE = 200.0
NORM = "group"
NORM_COLOR = "group"
NUM_STACK = 4
NUM_HOURGLASS = 2
SKIP_HOURGLASS = False
HG_DOWN = "ave_pool"
HOURGLASS_DIM = 256
MLP_DIM = [257, 1024, 512, 256, 128, 1]
MLP_DIM_COLOR = [513, 1024, 512, 256, 128, 3]
USE_TANH = False
RANDOM_FLIP = False
RANDOM_TRANS = False
RANDOM_SCALE = False
NO_RESIDUAL = False
SCHEDULE = [60, 80]
GAMMA = 0.1
COLOR_LOSS_TYPE = "l1"
VAL_TEST_ERROR = False
VAL_TRAIN_ERROR = False
GEN_TEST_MESH = False
GEN_TRAIN_MESH = False
ALL_MESH = False
NUM_GEN_MESH_TEST = 1
CHECKPOINTS_PATH = "./checkpoints"
LOAD_NETG_CHECKPOINT_PATH = "./checkpoints/net_G"
LOAD_NETC_CHECKPOINT_PATH = "./checkpoints/net_C"
RESULTS_PATH = "./results"
LOAD_CHECKPOINT_PATH = None
SINGLE = ""
MASK_PATH = None
IMG_PATH = None
AUG_ALSTD = 0.0
AUG_BRI = 0.0
AUG_CON = 0.0
AUG_SAT = 0.0
AUG_HUE = 0.0
AUG_BLUR = 0.0

# %%
def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf

# %%
def eval_grid_octree(coords, eval_func,init_resolution=64, threshold=0.01,num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)

# %%
def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)

# %%
def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix

# %%
def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


# %%
def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()

# %%
def gen_mesh(net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            net,
            cuda,
            calib_tensor,
            RESOLUTION,
            b_min,
            b_max,
            use_octree=use_octree
        )
            
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

# %%
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

# %%
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# %%
def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


# %%
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
  

# %%
class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

# class HGFilter(nn.Module):
    # def __init__(self):
    #     super(HGFilter, self).__init__()
    #     self.num_modules = NUM_STACK

    #     # Base part
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

    #     if NORM == 'batch':
    #         self.bn1 = nn.BatchNorm2d(64)
    #     elif NORM == 'group':
    #         self.bn1 = nn.GroupNorm(32, 64)

    #     if HG_DOWN == 'conv64':
    #         self.conv2 = ConvBlock(64, 64, NORM)
    #         self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    #     elif HG_DOWN == 'conv128':
    #         self.conv2 = ConvBlock(64, 128, NORM)
    #         self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
    #     elif HG_DOWN == 'ave_pool':
    #         self.conv2 = ConvBlock(64, 128, NORM)
    #     else:
    #         raise NameError('Unknown Fan Filter setting!')

    #     self.conv3 = ConvBlock(128, 128, NORM)
    #     self.conv4 = ConvBlock(128, 256, NORM)

    #     # Stacking part
    #     for hg_module in range(self.num_modules):
    #         self.add_module('m' + str(hg_module), HourGlass(1, NUM_HOURGLASS, 256, NORM))

    #         self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, NORM))
    #         self.add_module('conv_last' + str(hg_module),
    #                         nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
    #         if NORM == 'batch':
    #             self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
    #         elif NORM == 'group':
    #             self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))
                
    #         self.add_module('l' + str(hg_module), nn.Conv2d(256,
    #                                                         HOURGLASS_DIM, kernel_size=1, stride=1, padding=0))

    #         if hg_module < self.num_modules - 1:
    #             self.add_module(
    #                 'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
    #             self.add_module('al' + str(hg_module), nn.Conv2d(HOURGLASS_DIM,
    #                                                              256, kernel_size=1, stride=1, padding=0))

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)), True)
    #     tmpx = x
    #     if HG_DOWN == 'ave_pool':
    #         x = F.avg_pool2d(self.conv2(x), 2, stride=2)
    #     elif HG_DOWN in ['conv64', 'conv128']:
    #         x = self.conv2(x)
    #         x = self.down_conv2(x)
    #     else:
    #         raise NameError('Unknown Fan Filter setting!')

    #     normx = x

    #     x = self.conv3(x)
    #     x = self.conv4(x)

    #     previous = x

    #     outputs = []
    #     for i in range(self.num_modules):
    #         hg = self._modules['m' + str(i)](previous)

    #         ll = hg
    #         ll = self._modules['top_m_' + str(i)](ll)

    #         ll = F.relu(self._modules['bn_end' + str(i)]
    #                     (self._modules['conv_last' + str(i)](ll)), True)

    #         # Predict heatmaps
    #         tmp_out = self._modules['l' + str(i)](ll)
    #         outputs.append(tmp_out)

    #         if i < self.num_modules - 1:
    #             ll = self._modules['bl' + str(i)](ll)
    #             tmp_out_ = self._modules['al' + str(i)](tmp_out)
    #             previous = previous + ll + tmp_out_

    #     return outputs, tmpx.detach(), normx

# %%
class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y

# %%
class HGPIFuNet(nn.Module):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''
    def __init__(self,projection_mode='orthogonal',error_term=nn.MSELoss()):
        super(HGPIFuNet, self).__init__()
        self.name = 'hgpifu'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

        self.num_views = NUM_VIEWS

        vino_core = Core()
        # Replace with OpenVINO model
        # self.image_filter = HGFilter()
        self.image_filter = vino_core.compile_model(vino_core.read_model('./OV_model/FP16//HGFilter.xml'),"CPU")
        

        # Replace with OpenVINO model
        # self.surface_classifier = SurfaceClassifier(
        #     filter_channels=MLP_DIM,
        #     num_views=NUM_VIEWS,
        #     no_residual=NO_RESIDUAL,
        #     last_op=nn.Sigmoid()
        # )
        self.surface_classifier =  vino_core.compile_model(vino_core.read_model('./OV_model/FP16/SurfaceClassifier.xml').reshape("1, 256,?"), "CPU")
        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []
        
        # init_net(self)

    # replacing normalizer class member to class function.
    def normalizer(self, z, calibs=None, index_feat=None):
        z_feat = z * (LOADSIZE // 2) / Z_SIZE
        return z_feat

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        _, _, _, self.im_feat_list, self.tmpx, self.normx = self.image_filter.infer_new_request({0:images.cpu().numpy()}).values()
        self.im_feat_list = torch.tensor( self.im_feat_list).to("cuda")
        self.tmpx = torch.tensor( self.tmpx).to("cuda")
        self.normx = torch.tensor(self.normx).to("cuda")
        
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        

        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)

        if SKIP_HOURGLASS :
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if SKIP_HOURGLASS:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, points, calibs, transforms=None, labels=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        return res, error

# %%
class Evaluator:
    def __init__(self,  projection_mode='orthogonal'):
        self.load_size = LOADSIZE
        self.transforms = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        # cuda = torch.device('cpu')
        cuda = torch.device('cuda:0')

        # create net
        netG = HGPIFuNet(projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

       
        # netG.load_state_dict(torch.load(LOAD_NETG_CHECKPOINT_PATH, map_location=cuda),force=False)


        os.makedirs(RESULTS_PATH, exist_ok=True)
        os.makedirs('%s/%s' % (RESULTS_PATH, NAME), exist_ok=True)

        self.cuda = cuda
        self.netG = netG
        
    def to_tensor(self,x):
        return self.transforms(x)

    def load_image(self, image_path, mask_path):
        """
        這裡把 image 跟 mask 都讀進來，還有一些相機參數，最後把照片乘上 mask 這樣就只會有人的圖像而
        不會有背景
        """
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        # Return JSON 
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        with torch.no_grad():
            self.netG.eval()
            save_path = '%s/%s/result_%s.obj' % (RESULTS_PATH, NAME, data['name'])
            gen_mesh(self.netG, self.cuda, data,save_path=save_path, use_octree=use_octree)

# %%
evaluator = Evaluator()

test_images = glob.glob(os.path.join(TEST_FOLDER_PATH, '*'))
test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
test_masks = [f[:-4]+'_mask.png' for f in test_images]

for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
    try:
        """
        輸入準備
        """
        data = evaluator.load_image(image_path, mask_path)
        """
        Inference
        """
        evaluator.eval(data, True)
    except Exception as e:
        print("error:", e.args)

# %%



