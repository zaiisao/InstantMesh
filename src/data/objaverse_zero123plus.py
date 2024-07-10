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

from src.utils.train_util import instantiate_from_config


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
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


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
            vertices = vertices * scale
            
            # Recompute bounding box after scaling
            bbox_min = torch.min(vertices, dim=0).values
            bbox_max = torch.max(vertices, dim=0).values
            
            # Compute translation to center mesh at origin
            offset = -(bbox_min + bbox_max) / 2.0
            
            # Translate vertices
            vertices = vertices + offset
            
            return vertices
        
        mesh_vertices = normalize_mesh(mesh.vertices)
        mesh_faces = mesh.faces

        return mesh_vertices, mesh_faces

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
        mesh_vertices, mesh_faces = self.load_mesh(mesh_path)

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
            'target_depth_imgs': depths[1:],
            'mesh_vertices': mesh_vertices,
            'mesh_faces': mesh_faces
        }
        return data
