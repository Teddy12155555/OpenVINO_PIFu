import os

from PIL import Image
from skimage import measure
import numpy as np
import timeit
import torch
import torch.nn as nn
from torchvision import transforms

from openvino.runtime import Core, Tensor

NAME = "pifu_ov"
RESOLUTION = 256
TEST_FOLDER_PATH = "./input_images"
RESULTS_PATH = "./results"
Z_SIZE = 200.0
DEVICE = 'cpu'
OV_DEVICE = 'CPU'
HGFITER = './OV_model/FP16//HGFilter.xml'
SC = './OV_model/FP16/SurfaceClassifier.xml'


def load_image(image_path, mask_path):
    self_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
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
    mask = transforms.Resize(512)(mask)
    mask = transforms.ToTensor()(mask).float()
    # image
    image = Image.open(image_path).convert('RGB')
    image = self_transforms(image)
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

def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()

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

def eval_grid_octree(coords, eval_func,init_resolution=64, threshold=0.01,num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        test_mask = np.logical_and(grid_mask, dirty)
        points = coords[:, test_mask]
        
        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        
        dirty[test_mask] = False
        
        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
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

def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1])):
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000):
    
    coords, mat = create_grid(resolution, resolution, resolution,b_min, b_max)
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
        #return pred

    # Then we evaluate the grid
    
    sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)

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

def index(feat, uv):
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def orthogonal(points, calibrations):
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    return pts

def gen_mesh(net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)
    b_min = data['b_min']
    b_max = data['b_max']
    
    try:
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

class HGPIFuNet(nn.Module):
    def __init__(self):
        super(HGPIFuNet, self).__init__()
        self.name = 'hgpifu'
        
        self.index = index
        self.projection = orthogonal

        self.preds = None
        self.num_views = 1

        ##############
        #  OV field  #
        ##############
        self.core = Core()
        self.device_name = OV_DEVICE
        self.HGF_path = HGFITER
        self.SC_path = SC
        self.OV_int()
        
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []
        
        self.time_benchmark = 0

    def OV_int(self):
        HGF = self.core.read_model(self.HGF_path)
        self.HGF = self.core.compile_model(HGF, self.device_name)
        
        SC = self.core.read_model(self.SC_path)
        SC.reshape("1, 257, ?")
        self.SC = self.core.compile_model(SC, self.device_name)
    # replacing normalizer class member to class function.
    def normalizer(self, z):
        z_feat = z * (512 // 2) / Z_SIZE
        return z_feat

    def filter(self, images): 
        st = timeit.default_timer()
        input_tensor = images.cpu().numpy()
        self.time_benchmark += timeit.default_timer() - st
        
        results = self.HGF.infer_new_request({0: input_tensor})
        
        st = timeit.default_timer()
        for i, (k, v) in enumerate(results.items()):
            if i == 3:
                temp = torch.tensor(v).to(DEVICE)
            if i == 4:
                self.tmpx = torch.tensor(v).to(DEVICE)
            if i == 5:
                self.normx = torch.tensor(v).to(DEVICE)
        
        self.time_benchmark += timeit.default_timer() - st

        self.im_feat_list = [temp]
               
    def query(self, points, calibs):
        xyz = self.projection(points, calibs)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        
        z_feat = self.normalizer(z)

        self.intermediate_preds_list = []
        infer_request = self.SC.create_infer_request()
       
        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            
            point_local_feat = torch.cat(point_local_feat_list, 1)
            
            dy_shape = point_local_feat.shape[2]

            input_tensor_shape = Tensor(self.SC.input().element_type, [1, 257, dy_shape])
            infer_request.set_input_tensor(input_tensor_shape)
            
            st = timeit.default_timer()
            input_tensor = point_local_feat.cpu().numpy()
            self.time_benchmark += timeit.default_timer() - st
            
            infer_request.infer([input_tensor])
            result = infer_request.get_output_tensor()
            
            st = timeit.default_timer()
            result_ten = torch.Tensor(result.data[:]).to(DEVICE)
            self.time_benchmark += timeit.default_timer() - st
            
            pred = in_img[:,None].float() * result_ten
            
            
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_im_feat(self):
        return self.im_feat_list[-1]

    def get_preds(self):
        return self.preds

if __name__ == '__main__':
    IMAGE_PATH = './input_images\\ryota.png'
    MASK_PATH = './input_images\\ryota_mask.png'

    data = load_image(IMAGE_PATH, MASK_PATH)

    netG = HGPIFuNet().to(device=DEVICE)

    save_path = '%s/%s/result_notebook_version_%s.obj' % (RESULTS_PATH, NAME, data['name'])
    st = timeit.default_timer()
    gen_mesh(netG, DEVICE, data, save_path=save_path, use_octree=True)
    print('Time Cost: ', timeit.default_timer() - st)