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
from pathlib import Path

from src.utils.train_util import instantiate_from_config
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from .pipeline import RefOnlyNoisedUNet

from torch_scatter import scatter_max

from src.utils.camera_util import (
    get_zero123plus_angles
)

from .render_models.textured_mesh import TexturedMeshModel

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

    # def init_mesh_model(self, shape_path) -> nn.Module:
    #     fovyangle = np.pi / 3
    #     cache_path = Path('cache') / Path('shapes/spot_triangulated.obj').stem
    #     cache_path.mkdir(parents=True, exist_ok=True)
    #     model = TexturedMeshModel(
    #         # JA: GuideConfig values START
    #         dy=0.25,
    #         shape_scale=0.6,
    #         initial_texture=None,
    #         texture_interpolation_mode='bilinear',
    #         reference_texture=None,
    #         shape_path='shapes/spot_triangulated.obj',
    #         # JA: GuideConfig values END

    #         device=torch.device(f'cuda'),
    #         render_grid_size=1200,
    #         cache_path=cache_path,
    #         texture_resolution=1024,
    #         augmentations=False,
    #         fovyangle=fovyangle
    #     )

    #     model = model.to(self.device)

    #     return model
    
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

        mesh_vertices = batch['mesh_vertices'][:, None].repeat(1, batch['target_depth_imgs'].shape[1], 1, 1)
        mesh_faces = batch['mesh_faces']
        mesh_uvs = batch['mesh_uvs']
        mesh_face_uvs_idx = batch['mesh_face_uvs_idx']

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

        texture_img = nn.Parameter(torch.ones(1, 3, 512, 512).cuda() * 0.5)

        # azimuths = [0, 30, 90, 150, 210, 270, 330]
        # elevations = [0, 20, -10, 20, -10, 20, -10]
        azimuths, elevations = get_zero123plus_angles()
        azimuths = torch.from_numpy(azimuths).to(self.device, torch.float32)
        elevations = torch.from_numpy(elevations).to(self.device, torch.float32)

        def get_camera_from_multiple_view(elev, azim, r):
            # Adjust azimuth angle to account for coordinate system differences
            azim = azim + torch.pi

            # Calculate camera position using Blender's logic adjusted for Kaolin
            x = r * torch.sin(elev) * torch.sin(azim)
            y = r * torch.cos(elev)
            z = r * torch.sin(elev) * torch.cos(azim)

            pos = torch.stack([x, y, z], dim=1)
            look_at = torch.zeros_like(pos)
            camera_up_direction = torch.ones_like(pos) * torch.tensor([0.0, 1.0, 0.0]).to(pos.device)

            camera_proj = kaolin.render.camera.generate_transformation_matrix(pos, look_at, camera_up_direction)
            return camera_proj

        def normalize_multiple_depth(depth_maps):
            # assert (depth_maps.amax(dim=(1, 2)) <= 0).all(), 'depth map should be negative'
            assert not (depth_maps == 0).all(), 'depth map should not be empty'
            object_mask = depth_maps != 0  # Mask for non-background pixels

            # To handle operations for masked regions, we need to use masked operations
            # Set default min and max values to avoid affecting the normalization
            masked_depth_maps = torch.where(object_mask, depth_maps, torch.tensor(float('inf')).to(depth_maps.device))
            min_depth = masked_depth_maps.amin(dim=(1, 2), keepdim=True)

            masked_depth_maps = torch.where(object_mask, depth_maps, torch.tensor(-float('inf')).to(depth_maps.device))
            max_depth = masked_depth_maps.amax(dim=(1, 2), keepdim=True)

            range_depth = max_depth - min_depth

            # Calculate normalized depth maps
            min_val = 0.5
            normalized_depth_maps = torch.where(
                object_mask,
                ((1 - min_val) * (depth_maps - min_depth) / range_depth) + min_val,
                depth_maps # JA: Where the object mask is 0, depth map is 0 and we will return it
            )

            return normalized_depth_maps

        camera_transform = get_camera_from_multiple_view(
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

        # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
        depth_map_unnormalized_bhwc, _ = kaolin.render.mesh.rasterize(512, 512, face_vertices_camera[:, :, :, -1],
                                                            face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
        depth_map_unnormalized = depth_map_unnormalized_bhwc.permute(0, 3, 1, 2)
        depth_map = normalize_multiple_depth(depth_map_unnormalized)

        face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            mesh_uvs.repeat(6, 1, 1), #MJ: shape = (batch_size}, num_points, knum) =(1,4839,2)
            mesh_face_uvs_idx[0].long()        #MJ: shape = num_faces,face_size)=(7500,3)
        ).detach()

        uv_features, face_idx = kaolin.render.mesh.rasterize(512, 512, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes) # JA: https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.rasterize
        uv_features = uv_features.detach() #.permute(0, 3, 1, 2)

        mask = (face_idx > -1).float()[..., None]
        
        def create_face_view_map(face_idx):
            num_views, H, W = face_idx.shape

            # Flatten the face_idx tensor to make it easier to work with
            face_idx_flattened_2d = face_idx.view(num_views, -1)  # Shape becomes (num_views, H*W)

            # Get the indices of all elements
            # JA: From ChatGPT:
            # torch.meshgrid is used to create a grid of indices that corresponds to each dimension of the input tensor,
            # specifically in this context for the view indices and pixel indices. It allows us to pair each view index
            # with every pixel index, thereby creating a full coordinate system that can be mapped directly to the values
            # in the tensor face_idx.
            view_by_pixel_indices, pixel_by_view_indices = torch.meshgrid(
                torch.arange(num_views, device=face_idx.device),
                torch.arange(H * W, device=face_idx.device),
                indexing='ij'
            )

            # Flatten indices tensors
            view_by_pixel_indices_flattened = view_by_pixel_indices.flatten()
            pixel_by_view_indices_flattened = pixel_by_view_indices.flatten()

            faces_idx_view_pixel_flattened = face_idx_flattened_2d.flatten()

            # Convert pixel indices back to 2D indices (i, j)
            pixel_i_indices = pixel_by_view_indices_flattened // W
            pixel_j_indices = pixel_by_view_indices_flattened % W

            # JA: The original face view map is made of nested dictionaries, which is very inefficient. Face map information
            # is implemented as a single tensor which is efficient. Only tensors can be processed in GPU; dictionaries cannot
            # be processed in GPU.
            # The combined tensor represents, for each pixel (i, j), its view_idx 
            combined_tensor_for_face_view_map = torch.stack([
                faces_idx_view_pixel_flattened,
                view_by_pixel_indices_flattened,
                pixel_i_indices,
                pixel_j_indices
            ], dim=1)

            # Filter valid faces
            faces_idx_valid_mask = faces_idx_view_pixel_flattened >= 0

            # JA:
            # [[face_id_1, view_1, i_1, j_1]
            #  [face_id_1, view_1, i_2, j_2]
            #  [face_id_1, view_1, i_3, j_3]
            #  [face_id_1, view_2, i_4, j_4]
            #  [face_id_1, view_2, i_5, j_5]
            #  ...
            #  [face_id_2, view_1, i_k, j_l]
            #  [face_id_2, view_1, i_{k + 1}, j_{l + 1}]
            #  [face_id_2, view_2, i_{k + 2}, j_{l + 2}]]
            #  ...
            # The above example shows face_id_1 is projected, under view_1, to three pixels (i_1, j_1), (i_2, j_2), (i_3, j_3)
            # Shape is Nx4 where N is the number of pixels (no greater than H*W*num_views = 1200*1200*7) that projects the
            # valid face ID.
            return combined_tensor_for_face_view_map[faces_idx_valid_mask]

        def compare_face_normals_between_views(face_view_map, face_normals, face_idx):
            num_views, H, W = face_idx.shape
            weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool, device=face_idx.device)

            face_ids = face_view_map[:, 0] # JA: face_view_map.shape = (H*W*num_views, 4) = (1200*1200*7, 4) = (10080000, 4)
            views = face_view_map[:, 1]
            i_coords = face_view_map[:, 2]
            j_coords = face_view_map[:, 3]
            z_normals = face_normals[views, face_ids, 2] # JA: The shape of face_normals is (num_views, 3, num_faces)
                                                        # For example, face_normals can be (7, 3, 14232)
                                                        # z_normals is (N,)

            # Scatter z-normals into the tensor, ensuring each index only keeps the max value
            # JA: z_normals is the source/input tensor, and face_ids is the index tensor to scatter_max function.
            max_z_normals_over_views, _ = scatter_max(z_normals, face_ids, dim=0) # JA: N is a subset of length H*W*num_views
            # The shape of max_z_normals_over_N is the (num_faces,). The shape of the scatter_max output is equal to the
            # shape of the number of distinct indices in the index tensor face_ids.

            # Map the gathered max normals back to the respective face ID indices
            # JA: max_z_normals_over_views represents the max z normals over views for every face ID.
            # The shape of face_ids is (N,). Therefore the shape of max_z_normals_over_views_per_face is also (N,).
            max_z_normals_over_views_per_face = max_z_normals_over_views[face_ids]

            # Calculate the unworthy mask where current z-normals are less than the max per face ID
            unworthy_pixels_mask = z_normals < max_z_normals_over_views_per_face

            # JA: Update the weight masks. The shapes of face_view_map, whence views, i_coords, and j_coords were extracted
            # from, all have the shape of (N,), which represents the number of valid pixel entries. Therefore,
            # weight_masks[views, 0, i_coords, j_coords] will also have the shape of (N,) which allows the values in
            # weight_masks to be set in an elementwise manner.
            #
            # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
            # The above variable represents whether the pixel (i_coords[0], j_coords[0]) under views[0] is worthy to
            # contribute to the texture atlas.
            weight_masks[views, 0, i_coords, j_coords] = ~(unworthy_pixels_mask)

            # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
            # weight_masks[views[1], 0, i_coords[1], j_coords[1]] = ~(unworthy_pixels_mask[1])
            # weight_masks[views[2], 0, i_coords[2], j_coords[2]] = ~(unworthy_pixels_mask[2])
            # weight_masks[views[3], 0, i_coords[3], j_coords[2]] = ~(unworthy_pixels_mask[3])

            return weight_masks

        face_view_map = create_face_view_map(face_idx)
        weight_masks = compare_face_normals_between_views(face_view_map, face_normals, face_idx)

        def project_back(self, render_cache, background, rgb_output, object_mask):
            optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                        eps=1e-15)
            for _ in tqdm(range(200), desc='fitting mesh colors'):
                optimizer.zero_grad()
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)
                rgb_render = outputs['image']

                mask = object_mask.flatten()
                masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
                masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
                masked_mask = mask[mask > 0]
                loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()
                loss.backward()
                optimizer.step()

            return rgb_render

        # texture_map = texture_img.expand(6, -1, -1, -1)
        import torchvision
        texture_map =  torchvision.transforms.v2.functional.to_dtype(torchvision.io.read_image("/home/jahn/test.png").cuda())[None].expand(6, -1, -1, -1)
        image_features = kaolin.render.mesh.texture_mapping(uv_features, texture_map, mode="bilinear")
        image_features = image_features * mask

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
