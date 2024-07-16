import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import kaolin
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from .pipeline import RefOnlyNoisedUNet
from .renderer import Renderer

from torch_scatter import scatter_max

from src.utils.camera_util import (
    get_zero123plus_angles
)

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVDiffusion(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        drop_cond_prob=0.1,
        use_depth_controlnet=False,
    ):
        super(MVDiffusion, self).__init__()

        self.drop_cond_prob = drop_cond_prob

        self.register_schedule()

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )

        if use_depth_controlnet:
            pipeline.add_controlnet(ControlNetModel.from_pretrained(
                "sudo-ai/controlnet-zp11-depth-v1"
            ), conditioning_scale=0.75)

        self.pipeline = pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(self.pipeline.unet, train_sched, self.pipeline.scheduler)

        self.train_scheduler = train_sched      # use ddpm scheduler during training

        self.unet = pipeline.unet

        # self.mesh_model = None #self.init_mesh_model()

        # validation output buffer
        self.validation_step_outputs = []

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())

    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.pipeline.to(device)

        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch['cond_imgs']      # (B, C, H, W)
        cond_imgs = cond_imgs.to(self.device)

        # random resize the condition image
        cond_size = np.random.randint(128, 513)
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)

        target_imgs = batch['target_imgs']  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(target_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
        target_imgs = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)    # (B, C, 3H, 2W)
        target_imgs = target_imgs.to(self.device)

        target_depth_imgs = batch['target_depth_imgs']  # (B, 6, C, H, W)
        target_depth_imgs = v2.functional.resize(target_depth_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
        target_depth_imgs = rearrange(target_depth_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)    # (B, C, 3H, 2W)
        target_depth_imgs = target_depth_imgs.to(self.device)

        num_viewpoints = target_imgs.shape[1]

        padded_mesh_vertices = batch['padded_mesh_vertices']#[:, None].repeat(1, num_viewpoints, 1, 1)
        padded_mesh_faces = batch['padded_mesh_faces']
        padded_mesh_uvs = batch['padded_mesh_uvs']
        padded_mesh_face_uvs_idx = batch['padded_mesh_face_uvs_idx']

        def unpad_tensors(padded_tensors, pad_value=-1):
            unpadded_tensors = []
            
            for tensor in padded_tensors:
                # Find the length of valid data by looking for the first occurrence of the pad_value
                valid_length = (tensor != pad_value).all(dim=1).nonzero(as_tuple=True)[0].max().item() + 1
                # Slice the tensor up to the valid length
                unpadded_tensor = tensor[:valid_length]
                unpadded_tensors.append(unpadded_tensor)
            
            return unpadded_tensors
        
        mesh_vertices = unpad_tensors(padded_mesh_vertices)
        mesh_faces = unpad_tensors(padded_mesh_faces)
        mesh_uvs = unpad_tensors(padded_mesh_uvs)
        mesh_face_uvs_idx = unpad_tensors(padded_mesh_face_uvs_idx)

        return cond_imgs, target_imgs, target_depth_imgs, mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx
    
    @torch.no_grad()
    def forward_vision_encoder(self, images):
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_clip(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        global_embeds = self.pipeline.vision_encoder(image_pt, output_hidden_states=False).image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        encoder_hidden_states = self.pipeline._encode_prompt("", self.device, 1, False)[0]
        ramp = global_embeds.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states
    
    @torch.no_grad()
    def encode_condition_image(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_vae(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents
    
    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    
    def forward_unet(self, latents, t, prompt_embeds, cond_latents, control_depth):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        control_depth = control_depth.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents, control_depth=control_depth)
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def compute_seam_loss(self, mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx):
        texture_img = nn.Parameter(torch.ones(1, 3, 512, 512).cuda() * 0.5)

        # azimuths = [0, 30, 90, 150, 210, 270, 330]
        # elevations = [0, 20, -10, 20, -10, 20, -10]
        azimuths, elevations = get_zero123plus_angles()
        azimuths = torch.from_numpy(azimuths).to(self.device, torch.float32)
        elevations = torch.from_numpy(elevations).to(self.device, torch.float32)

        mesh_renderer = Renderer(mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx)

        camera_transform = mesh_renderer.get_camera_from_views(
            torch.deg2rad(90 - elevations),
            torch.deg2rad(90 + azimuths),
            r=1.5
        )

        sensor_width = 32
        focal_length = 35
        fovyangle = 2 * math.atan(sensor_width / (2 * focal_length))

        camera_projection = kaolin.render.camera.generate_perspective_projection(fovyangle).to(self.device)

        face_vertices_camera_list, face_vertices_image_list, face_normals_list = [], [], []
        for batch_num in range(B):
            face_vertices_camera_one_mesh, face_vertices_image_one_mesh, face_normals_one_mesh = \
                kaolin.render.mesh.prepare_vertices(
                    mesh_vertices[batch_num], mesh_faces[batch_num],
                    camera_projection,
                    camera_transform=camera_transform
                )

            face_vertices_camera_list.append(face_vertices_camera_one_mesh)
            face_vertices_image_list.append(face_vertices_image_one_mesh)
            face_normals_list.append(face_normals_one_mesh)

        face_vertices_camera = torch.cat(face_vertices_camera_list, dim=0)
        face_vertices_image = torch.cat(face_vertices_image_list, dim=0)
        face_normals = torch.cat(face_normals_list, dim=0)

        face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            mesh_uvs.repeat(6, 1, 1), #MJ: shape = (batch_size}, num_points, knum) =(1,4839,2)
            mesh_face_uvs_idx[0].long()        #MJ: shape = num_faces,face_size)=(7500,3)
        ).detach()

        uv_features, face_idx = kaolin.render.mesh.rasterize(512, 512, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes) # JA: https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.rasterize
        uv_features = uv_features.detach() #.permute(0, 3, 1, 2)

        mask = (face_idx > -1).float()[..., None]

        face_view_map = mesh_renderer.create_face_view_map(face_idx)
        weight_masks = mesh_renderer.compare_face_normals_between_views(face_view_map, face_normals, face_idx)

        texture_map = texture_img.expand(6, -1, -1, -1)
        image_features = kaolin.render.mesh.texture_mapping(uv_features, texture_map, mode="bilinear")
        image_features = image_features * mask
    
    def training_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs, target_depth_imgs, \
        mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx = self.prepare_batch_data(batch)

        # sample random timestep
        B = cond_imgs.shape[0]
        
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            prompt_embeds = self.pipeline._encode_prompt([""]*B, self.device, 1, False)
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs)
            cond_latents = self.encode_condition_image(cond_imgs)

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        
        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, cond_latents, target_depth_imgs)
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)

        seam_loss = self.compute_seam_loss(mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx)

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.global_step % 500 == 0 and self.global_rank == 0:
            with torch.no_grad():
                latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)

                latents = unscale_latents(latents_pred)
                images = unscale_image(self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
                images = (images * 0.5 + 0.5).clamp(0, 1)
                images = torch.cat([target_imgs, images], dim=-2)

                grid = make_grid(images, nrow=images.shape[0], normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png'))

        return loss
        
    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs = self.prepare_batch_data(batch)

        images_pil = [v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])]

        outputs = []
        for cond_img in images_pil:
            latent = self.pipeline(cond_img, num_inference_steps=75, output_type='latent').images
            image = unscale_image(self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
            image = (image * 0.5 + 0.5).clamp(0, 1)
            outputs.append(image)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target_imgs, outputs], dim=-2)
        
        self.validation_step_outputs.append(images)
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(grid, os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png'))

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/4)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
