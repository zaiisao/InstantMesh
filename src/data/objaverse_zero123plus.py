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
from typing import List, Tuple, Optional

import struct

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

def collate_fn(data: List[Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    str,
    int
]]):
    cond_imgs, target_imgs, target_depth_img, \
        mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx, image_path, cond_azimuths = zip(*data)

    data = {
        'cond_imgs': torch.stack(cond_imgs),
        'target_imgs': torch.stack(target_imgs),
        'target_depth_imgs': torch.stack(target_depth_img),

        'mesh_vertices': mesh_vertices,
        'mesh_faces': mesh_faces,
        'mesh_uvs': mesh_uvs,
        'mesh_face_uvs_idx': mesh_face_uvs_idx,

        'image_path': image_path,

        'cond_azimuths': cond_azimuths
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
        return wds.WebLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(
            self.datasets['validation'],
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return wds.WebLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )


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
    
    def load_mesh(self, image_path):
        mesh_vertices = torch.load(os.path.join(image_path, 'mesh_vertices.pt'))
        mesh_faces = torch.load(os.path.join(image_path, 'mesh_faces.pt'))
        mesh_uvs = torch.load(os.path.join(image_path, 'mesh_uvs.pt'))
        mesh_face_uvs_idx = torch.load(os.path.join(image_path, 'mesh_face_uvs_idx.pt'))

        with open(os.path.join(image_path, 'azimuth.bin'), 'rb') as file:
            binary_data = file.read(2)
            cond_azimuth = struct.unpack('h', binary_data)[0]

        return mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx, cond_azimuth

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

        # JA: The modified dataset format includes the .glb file in each folder
        # mesh_path = os.path.join(image_path, f'{self.paths[index]}.glb')
        try:
            mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx, cond_azimuth = self.load_mesh(image_path)
        except:
            # JA: Some meshes do not include UVs. These should be skipped for the purposes of use seam loss
            mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx, cond_azimuth = None, None, None, None, None

        # Commented by JA: Including the mesh vertices, faces, etc. here will not work because they have varying lengths.
        # This requires us to return the values themselves, so that they can be handled in the collate_fn with the usage
        # of padding the shorter tensors to match the length of the longest tensor

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

        # print(image_path, cond_imgs, target_imgs, target_depth_imgs, mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx)

        return \
            cond_imgs, target_imgs, target_depth_imgs, \
            mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx, \
            image_path, \
            cond_azimuth
