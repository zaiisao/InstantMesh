import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
import kaolin
from typing import List, Tuple

from src.utils.train_util import instantiate_from_config

def pad_tensors(tensor_list, pad_value=-1):
    max_len = max(tensor.shape[0] for tensor in tensor_list)
    padded_tensors = []

    for tensor in tensor_list:
        pad_size = max_len - tensor.shape[0]

        # Create a padding tensor with the same type and device as the original tensor
        pad_tensor = torch.full((pad_size, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)

        # Concatenate the original tensor with the padding tensor
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)

        padded_tensors.append(padded_tensor)

    return padded_tensors

def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    cond_imgs, target_imgs, target_depth_img, mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx = zip(*data)

    padded_mesh_vertices = pad_tensors(mesh_vertices)
    padded_mesh_faces = pad_tensors(mesh_faces)
    padded_mesh_uvs = pad_tensors(mesh_uvs)
    padded_mesh_face_uvs_idx = pad_tensors(mesh_face_uvs_idx)

    data = {
        'cond_imgs': torch.stack(cond_imgs),
        'target_imgs': torch.stack(target_imgs),
        'target_depth_imgs': torch.stack(target_depth_img),

        'padded_mesh_vertices': torch.stack(padded_mesh_vertices),
        'padded_mesh_faces': torch.stack(padded_mesh_faces),
        'padded_mesh_uvs': torch.stack(padded_mesh_uvs),
        'padded_mesh_face_uvs_idx': torch.stack(padded_mesh_face_uvs_idx)
    }

    return data

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler, collate_fn=collate_fn)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler, collate_fn=collate_fn)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir

        with open(os.path.join(root_dir, meta_fname)) as f:
            lvis_dict = json.load(f)
        paths = []
        for k in lvis_dict.keys():
            paths.extend(lvis_dict[k])
        self.paths = paths
            
        # total_objects = len(self.paths)
        # if validation:
        #     self.paths = self.paths[-16:] # used last 16 as validation
        # else:
        #     self.paths = self.paths[:-16]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_mesh(self, mesh_path):
        mesh = kaolin.io.gltf.import_mesh(mesh_path)

        def normalize_mesh(vertices):
            # Compute bounding box
            bbox_min = torch.min(vertices, dim=0).values
            bbox_max = torch.max(vertices, dim=0).values
            
            # Compute scale factor to fit within unit cube
            scale = 1.0 / torch.max(bbox_max - bbox_min)
            
            # Scale vertices
            scaled_vertices = vertices * scale
            
            # Recompute bounding box after scaling
            bbox_min = torch.min(scaled_vertices, dim=0).values
            bbox_max = torch.max(scaled_vertices, dim=0).values

            # Compute translation to center mesh at origin
            bbox_center = (bbox_min + bbox_max) / 2.0

            # Translate vertices
            vertices_from_bbox_center = scaled_vertices - bbox_center

            # JA: scaled_vertices on the right-hand side refers to the coordiantes of the vertices from
            # the world coordinate system

            return vertices_from_bbox_center
        
        mesh_vertices = normalize_mesh(mesh.vertices)
        mesh_faces = mesh.faces
        mesh_uvs = mesh.uvs
        mesh_face_uvs_idx = mesh.face_uvs_idx

        return mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx

    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.image_dir, self.paths[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            depth_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    depth, alpha = self.load_im(os.path.join(image_path, '%03d_depth.png' % idx), bkg_color)
                    img_list.append(img)
                    depth_list.append(depth)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()
        depths = torch.stack(depth_list, dim=0).float()
        
        mesh_path = os.path.join(image_path, f'{self.paths[index]}.glb')
        mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx = self.load_mesh(mesh_path)

        # data = {
        #     'cond_imgs': imgs[0],           # (3, H, W)
        #     'target_imgs': imgs[1:],        # (6, 3, H, W)
        #     'target_depth_imgs': depths[1:],
        #     'mesh_vertices': mesh_vertices,
        #     'mesh_faces': mesh_faces,
        #     'mesh_uvs': mesh_uvs,
        #     'mesh_face_uvs_idx': mesh_face_uvs_idx
        # }

        cond_imgs = imgs[0]
        target_imgs = imgs[1:]
        target_depth_imgs = depths[1:]

        return cond_imgs, target_imgs, target_depth_imgs, mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx
