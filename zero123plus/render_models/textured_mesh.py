import os

import kaolin as kal
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from .mesh import Mesh
from .render import Renderer

def build_cotan_laplacian_torch(points_tensor: torch.Tensor, tris_tensor: torch.Tensor) -> np.ndarray:
    tris, points = tris_tensor.cpu().numpy(), points_tensor.cpu().numpy()

    a, b, c = (tris[:, 0], tris[:, 1], tris[:, 2])
    A = np.take(points, a, axis=1)
    B = np.take(points, b, axis=1)
    C = np.take(points, c, axis=1)

    eab, ebc, eca = (B - A, C - B, A - C)
    eab = eab / np.linalg.norm(eab, axis=0)[None, :]
    ebc = ebc / np.linalg.norm(ebc, axis=0)[None, :]
    eca = eca / np.linalg.norm(eca, axis=0)[None, :]

    alpha = np.arccos(-np.sum(eca * eab, axis=0))
    beta = np.arccos(-np.sum(eab * ebc, axis=0))
    gamma = np.arccos(-np.sum(ebc * eca, axis=0))

    wab, wbc, wca = (1.0 / np.tan(gamma), 1.0 / np.tan(alpha), 1.0 / np.tan(beta))
    rows = np.concatenate((a, b, a, b, b, c, b, c, c, a, c, a), axis=0)
    cols = np.concatenate((a, b, b, a, b, c, c, b, c, a, a, c), axis=0)
    vals = np.concatenate((wab, wab, -wab, -wab, wbc, wbc, -wbc, -wbc, wca, wca, -wca, -wca), axis=0)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(points.shape[1], points.shape[1]), dtype=float).tocsc()
    return L


def build_graph_laplacian_torch(tris_tensor: torch.Tensor) -> np.ndarray:
    tris = tris_tensor.cpu().numpy()
    n_verts = tris.max() + 1
    v2v = [[] for _ in range(n_verts)]
    for face in tris:
        for i in face:
            for j in face:
                if i != j:
                    if j not in v2v[i]:
                        v2v[i].append(j)

    valency = [len(x) for x in v2v]
    I, J, vals = [], [], []
    for i in range(n_verts):
        I.append(i)
        J.append(i)
        vals.append(1)

        for neighbor in v2v[i]:
            I.append(i)
            J.append(neighbor)
            vals.append(-1 / valency[i])
    L = sparse.csr_matrix((vals, (I, J)), shape=(n_verts, n_verts))
    return L


def eigen_problem(Lap, k=20, e=0.0) -> (torch.Tensor, torch.Tensor):
    shift = 1e-4
    eigenvalues, eigenvectors = eigsh(
        Lap + shift * scipy.sparse.eye(Lap.shape[0]),
        k=k + 1, which='LM', sigma=e, tol=1e-3)
    eigenvalues += shift

    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    return torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors.T).float()


def choose_multi_modal(n: int, k: int):
    interval_length = n // k
    n_intervals = n / interval_length
    if n_intervals == int(n_intervals):
        n_intervals = int(n_intervals)
    else:
        n_intervals = int(n_intervals) + 1
    chosen_numbers = []
    for i in range(n_intervals):
        current_interval_length = min(interval_length, n - i * interval_length)
        chosen_numbers.append(np.random.choice(current_interval_length) + i * interval_length)
    return chosen_numbers


class TexturedMeshModel(nn.Module):
    def __init__(self,
                 dy=0.25,
                 shape_scale=0.6,
                 initial_texture=None,
                 texture_interpolation_mode='bilinear',
                 reference_texture=None,
                 shape_path='shapes/spot_triangulated.obj',
                 render_grid_size=1024,
                 texture_resolution=1024,
                 initial_texture_path=None,
                 cache_path=None,
                 device=torch.device('cpu'),
                 augmentations=False,
                 augment_prob=0.5,
                 fovyangle=np.pi / 3):

        super().__init__()
        self.device = device
        self.augmentations = augmentations
        self.augment_prob = augment_prob
        self.dy = dy
        self.mesh_scale = shape_scale
        self.shape_path = shape_path
        self.texture_resolution = texture_resolution
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = initial_texture
        self.cache_path = cache_path
        self.num_features = 3

        self.dim = (render_grid_size, render_grid_size)

        self.renderer = Renderer(device=self.device, dim=self.dim,
                                 interpolation_mode=texture_interpolation_mode, fovyangle=fovyangle)
        self.env_sphere, self.mesh = self.init_meshes()
        self.default_color = [0.8, 0.1, 0.8] # JA: This is the magenta color, set to the texture atlas
        self.background_sphere_colors, self.texture_img = self.init_paint() # JA: self.texture_img is a learnable parameter
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img)) # JA: self.texture_img is the texture atlas
                                # define self.meta_texture_img variable to be the parameter of the neural network whose init
                                # value is set to a zero tensor
        if reference_texture: # JA: This value is None by default
            base_texture = torch.Tensor(np.array(Image.open(reference_texture).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
            change_mask = (
                    (base_texture.to(self.device) - self.texture_img).abs().sum(axis=1) > 0.1).float()
            with torch.no_grad():
                self.meta_texture_img[:, 1] = change_mask
        self.vt, self.ft = self.init_texture_map()

        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long()).detach()

        self.n_eigen_values = 20
        self._L = None
        self._eigenvalues = None
        self._eigenvectors = None
   
    def render_face_normals_face_idx(self, verts, faces, uv_face_attr, elev, azim, radius,
                                   look_at_height=0.0, dims=None, background_type='none'):
        dims = self.dim if dims is None else dims
       
        camera_transform = self.renderer.get_camera_from_multiple_view(
            elev, azim, r=radius,
            look_at_height=look_at_height
        )
        # JA: Since the function prepare_vertices is specifically designed to move and project vertices to camera space and then index them with faces, the face normals returned by this function are also relative to the camera coordinate system. This follows from the context provided by the documentation, where the operations involve transforming vertices into camera coordinates, suggesting that the normals are computed in relation to these transformed vertices and thus would also be in camera coordinates.
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts, faces, self.renderer.camera_projection, camera_transform=camera_transform)
        # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
        depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                            face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
        depth_map = self.renderer.normalize_multiple_depth(depth_map)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, uv_face_attr)
        uv_features = uv_features.detach()
    
        mask = (face_idx > -1).float()[..., None] #MJ: face_idx: (7,1200,1200); mask: (7,1200,1200,1) => (7,1,1200,1200) by  mask.permute(0, 3, 1, 2

        # JA: face_normals[0].shape:[14232, 3], face_idx.shape: [1, 1024, 1024]
        # normals_image = face_normals[0][face_idx, :] # JA: normals_image: [1024, 1024, 3]
        # Generate batch indices
        batch_size = face_normals.shape[0]
        batch_indices = torch.arange(batch_size).view(-1, 1, 1)

        # Expand batch_indices to match the dimensions of face_idx
        batch_indices = batch_indices.expand(-1, *face_idx.shape[1:])

        # Use advanced indexing to gather the results
        normals_image = face_normals[batch_indices, face_idx]

       
        return mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), \
               face_normals.permute(0, 2, 1), face_idx[:, None, :, :]


    @property
    def L(self) -> np.ndarray:
        if self._L is None:
            self._L = build_cotan_laplacian_torch(self.mesh.vertices.T, self.mesh.faces)
        return self._L

    def eigens(self, k: int, e: float) -> (torch.Tensor, torch.Tensor):
        if self._eigenvalues is None or self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = eigen_problem(self.L, k, e)
            self._eigenvalues, self._eigenvectors = \
                self._eigenvalues.to(self.device), self._eigenvectors.to(self.device)

        return self._eigenvalues, self._eigenvectors

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
        vertices -= vertices.mean(dim=0)[None, :]
        vertices /= vertices.norm(dim=1).max()
        vertices *= mesh_scale
        vertices[:, 1] += dy
        return vertices

    def spectral_augmentations(self, vertices: torch.Tensor) -> torch.Tensor:
        eigen_values, basis_functions = self.eigens(self.n_eigen_values, 0.0)
        basis_functions /= basis_functions.max(dim=-1)[0][:, None] - basis_functions.min(dim=-1)[0][:, None]

        chosen_basis_function = choose_multi_modal(basis_functions.shape[0], 2)
        coeffs = torch.zeros(basis_functions.shape[0]).to(self.device)
        signs = ((torch.rand(len(chosen_basis_function), device=self.device) > 0.5).float() - 0.5) * 2
        coeffs[chosen_basis_function] = signs

        reconstructed = coeffs @ basis_functions.float()

        directions = vertices / torch.norm(vertices, dim=1)[:, None]
        deformed_v = vertices + 0.25 * reconstructed[:, None] * directions
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def axis_augmentations(self, vertices: torch.Tensor, stretch_factor: float = 1.6, squish_factor: float = 0.7):
        axis_indices = np.arange(0, 3)
        axis_indices = np.random.permutation(axis_indices)
        stretch_axis = axis_indices[0]
        squish_axis = axis_indices[1]

        deformed_v = vertices.clone()
        deformed_v[:, stretch_axis] *= stretch_factor
        deformed_v[:, squish_axis] *= squish_factor
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def augment_vertices(self):
        verts = self.mesh.vertices.clone()
        if np.random.rand() < 0.5:
            verts = self.spectral_augmentations(verts)
        if np.random.rand() < 0.5:
            verts = self.axis_augmentations(verts)
        return verts

    def init_meshes(self, env_sphere_path='shapes/env_sphere.obj'):
        env_sphere = Mesh(env_sphere_path, self.device)

        mesh = Mesh(self.shape_path, self.device)
        mesh.normalize_mesh(inplace=True, target_scale=self.mesh_scale, dy=self.dy) # JA: Normalize mesh into 1x1x1 cube.
                                                                                    # target_scale is 0.6, dy is 0.25

        return env_sphere, mesh

    def zero_meta(self):
        with torch.no_grad():
            self.meta_texture_img[:] = 0

    def init_paint(self, num_backgrounds=1):
        # random color face attributes for background sphere
        init_background_bases = torch.rand(num_backgrounds, 3).to(self.device)
        modulated_init_background_bases_latent = init_background_bases[:, None, None, :] * 0.8 + 0.2 * torch.randn(
            num_backgrounds, self.env_sphere.faces.shape[0],
            3, self.num_features, dtype=torch.float32).cuda()
        background_sphere_colors = nn.Parameter(modulated_init_background_bases_latent.cuda())

        if self.initial_texture_path is not None:
            texture = torch.Tensor(np.array(Image.open(self.initial_texture_path).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
        else: # JA: This is the case in our experiment
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).cuda() * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).cuda()
        texture_img = nn.Parameter(texture)
        return background_sphere_colors, texture_img

    def invert_color(self, color: torch.Tensor) -> torch.Tensor:
        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2

        pinv = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T)
        if len(color) == 1 or type(color) is torch.Tensor:
            init_color_in_latent = color @ pinv.T
        else:
            init_color_in_latent = pinv @ torch.tensor(
                list(color)).float().to(A.device)
        return init_color_in_latent

    def change_default_to_median(self):
        diff = (self.texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        default_mask = (diff < 0.1).float().unsqueeze(0)
        median_color = self.texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(axis=1)
        with torch.no_grad():
            self.texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)

    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        if self.mesh.vt is not None and self.mesh.ft is not None \
                and self.mesh.vt.shape[0] > 0 and self.mesh.ft.min() > -1:
            logger.info('Mesh includes UV map')
            vt = self.mesh.vt.cuda()
            ft = self.mesh.ft.cuda()
        elif cache_exists_flag:
            logger.info(f'running cached UV maps from {vt_cache}')
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
        else:
            logger.info(f'running xatlas to unwrap UVs for mesh')
            # unwrap uvs
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2] # JA: vt stores texture UV coordinates for each vertex. ft stores indices to the UV array for each face.

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        raise NotImplementedError
    
    def get_params(self):
        # return [self.background_sphere_colors, self.texture_img, self.meta_texture_img]
         return [self.texture_img, self.meta_texture_img]
          # JA: In our experiment, self.background_sphere_colors
          # are not used as parameters of the loss function


    def get_params_texture_atlas(self):
        # return [self.background_sphere_colors, self.texture_img, self.meta_texture_img]
        return [self.texture_img]    # JA: In our experiment, self.background_sphere_colors
                                                            # are not used as parameters of the loss function

    def get_params_max_z_normals(self):
         return [self.meta_texture_img]
        # return [self.texture_img]    # JA: In our experiment, self.background_sphere_colors
                                                            # are not used as parameters of the loss function

    @torch.no_grad()
    def export_mesh(self, path):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        h0, w0 = 256, 256
        ssaa, name = 1, ''

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        colors = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)

        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        colors = Image.fromarray(colors)

        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)

        colors.save(os.path.join(path, f'{name}albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')

        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info('writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {v[0]} {v[1]} \n')

            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    def render(self, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None, use_batch_render=True):
        if use_batch_render:
            if theta is not None:
                if isinstance(theta, float) or isinstance(theta, int):
                    theta = torch.tensor([theta]).to(self.device) # JA: [B,]
                elif isinstance(theta, list):
                    theta = torch.tensor(theta).to(self.device) # JA: [B,]

            if phi is not None:
                if isinstance(phi, float) or isinstance(phi, int):
                    phi = torch.tensor([phi]).to(self.device)
                elif isinstance(phi, list):
                    phi = torch.tensor(phi).to(self.device)

            if radius is not None:
                if isinstance(radius, float) or isinstance(radius, int):
                    radius = torch.tensor([radius]).to(self.device)
                elif isinstance(radius, list):
                    radius = torch.tensor(radius).to(self.device)

            return self.render_batch(theta, phi, radius, background,
                use_meta_texture, render_cache, use_median, dims)
        else:
            return self.render_legacy(theta, phi, radius, background,
                use_meta_texture, render_cache, use_median, dims)
        
    def render_legacy(self, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        background_sphere_colors = self.background_sphere_colors[
            torch.randint(0, self.background_sphere_colors.shape[0], (1,))]
        if use_meta_texture: # JA: During training of the network, we either learn meta texture or texture
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img  # JA: texture_img is the render image which is actually the
                                            # learnable texture atlas

        if self.augmentations:
            augmented_vertices = self.augment_vertices()
        else:
            augmented_vertices = self.mesh.vertices

        if use_median:
            diff = (texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)
            default_mask = (diff < 0.1).float().unsqueeze(0)
            median_color = texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)
            texture_img = texture_img.clone()
            with torch.no_grad():
                texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str: # JA: If background is a string, set it as the type
            background_type = background
            use_render_back = True
        pred_features, mask, depth, normals, render_cache = self.renderer.render_single_view_texture(augmented_vertices,
                                                                                                     self.mesh.faces,
                                                                                                     self.face_attributes,
                                                                                                     texture_img,
                                                                                                     elev=theta,
                                                                                                     azim=phi,
                                                                                                     radius=radius,
                                                                                                     look_at_height=self.dy,
                                                                                                     render_cache=render_cache,
                                                                                                     dims=dims,
                                                                                                     background_type=background_type)

        mask = mask.detach()

        if use_render_back:
            pred_map = pred_features
            pred_back = pred_features
        else:
            if background is None:
                # JA: background_sphere_colors is used when the background is None
                pred_back, _, _ = self.renderer.render_single_view(self.env_sphere,
                                                                   background_sphere_colors,
                                                                   elev=theta,
                                                                   azim=phi,
                                                                   radius=radius,
                                                                   dims=dims,
                                                                   look_at_height=self.dy, calc_depth=False)
            elif len(background.shape) == 1:
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:
                pred_back = background

            pred_map = pred_back * (1 - mask) + pred_features * mask

        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)

        return {'image': pred_map, 'mask': mask, 'background': pred_back,
                'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
                'texture_map': texture_img}
    
    def render_batch(self, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None):
        if render_cache is None:
            batch_size = theta.shape[0]
            assert theta is not None and phi is not None and radius is not None
        else:
            batch_size = render_cache["uv_features"].shape[0]
        background_sphere_colors = self.background_sphere_colors[
            torch.randint(0, self.background_sphere_colors.shape[0], (batch_size,))]
        texture_img = self.meta_texture_img if use_meta_texture else self.texture_img

        if self.augmentations:
            augmented_vertices = self.augment_vertices()
        else:
            augmented_vertices = self.mesh.vertices

        if use_median: #MJ: check if the texture_img being learned is not so different from the default magenta color
            diff = (texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)
            default_mask = (diff < 0.1).float().unsqueeze(0)
            median_color = texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)  #MJ: get the median color of the non-magenta region of texture_img
            texture_img = texture_img.clone()
            with torch.no_grad(): #MJ: fill the default (magenta) region of texture_img by the median color
                texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
                
        #MJ:  When rendering images, having large patches of a default or placeholder color (like magenta) 
        #  can be visually jarring and unrealistic. By filling these regions with a median color derived
        #  from the actual textured parts of the image, the overall appearance becomes more cohesive 
        #  and aesthetically pleasing. This helps in creating a more seamless and realistic image,
        #  especially in contexts  where the texture details are crucial, such as in photorealistic rendering.  
        
        #==>
        # Coverage Gaps: Despite the intention to cover the entire texture map with data from various 
        # viewpoints, gaps can occur. This might be due to occlusions, insufficient viewpoint coverage,
        # or limitations in the image processing pipeline (e.g., alignment errors or inadequate resolution). 
        # In such cases,  some regions of the texture map may not receive any data, resulting in default color patches.     
        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str: # JA: If background is a string, set it as the type
            background_type = background
            use_render_back = True
        
        # JA: We need to repeat several tensors to support the batch size.
        # For example, with a tensor of the shape, [1, 3, 1200, 1200], doing
        # repeat(batch_size, 1, 1, 1) results in [1 * batch_size, 3 * 1, 1200 * 1, 1200 * 1]
        pred_features, mask, depth, normals, render_cache = self.renderer.render_multiple_view_texture(
            augmented_vertices[None].repeat(batch_size, 1, 1),
            self.mesh.faces, # JA: the faces tensor can be shared across the batch and does not require its own batch dimension.
            self.face_attributes.repeat(batch_size, 1, 1, 1),
           #MJ :texture_img.repeat(batch_size, 1, 1, 1),
            texture_img.expand(batch_size, -1, -1, -1), #MJ: Note the use of -1 in .expand(), which tells PyTorch to keep the size of those dimensions as is (3, 1024, and 1024, respectively).
            elev=theta,
            azim=phi,
            radius=radius,
            look_at_height=self.dy,
            render_cache=render_cache,
            dims=dims,
            background_type=background_type
        )
        
        #   texture_img.repeat(batch_size, 1, 1, 1),: When you use the repeat method in PyTorch, such as texture_img.repeat(batch_size, 1, 1, 1), 
        # it does indeed create separate copies of texture_img in the resulting tensor, but these copies are
        # not independent in terms of gradient computation.
        # Gradients: Although repeat creates what seems like separate copies of the data for forward computation,
        # all these copies still reference back to the original texture_img tensor when it comes to computing gradients.
        # If a gradient is computed with respect to the repeated tensor during backpropagation, 
        # it will aggregate (sum) the gradients from all copies back into the original texture_img.
        # This means that the gradient of each parameter in texture_img will be the sum of the gradients
        # computed across all batch instances where it was used.
        
        # Using .expand() is particularly useful in models where a parameter tensor (like weights, biases,
        # or a specific feature map) needs to be applied identically across multiple instances 
        # or positions without creating multiple independent copies, thus saving memory and computational resources.
        

        mask = mask.detach()

        if use_render_back:
            pred_map = pred_features
            pred_back = pred_features
        else:
            if background is None:
                pred_back, _, _ = self.renderer.render_multiple_view(self.env_sphere,
                                                                   background_sphere_colors,
                                                                   elev=theta,
                                                                   azim=phi,
                                                                   radius=radius,
                                                                   dims=dims,
                                                                   look_at_height=self.dy, calc_depth=False)
            elif len(background.shape) == 1:
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:
                pred_back = background

            pred_map = pred_back * (1 - mask) + pred_features * mask

        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)

        return {'image': pred_map, 'mask': mask, 'background': pred_back,
                'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
                'texture_map': texture_img}

    def draw(self, theta, phi, radius, target_rgb):
        # failed attempt to draw on the texture image

        uv_features, face_idx = self.renderer.project_uv_single_view(self.mesh.vertices,
                                                                     self.mesh.faces,
                                                                     self.face_attributes,
                                                                     elev=theta,
                                                                     azim=phi,
                                                                     radius=radius,
                                                                     look_at_height=self.dy)
        unique_face_idx = torch.unique(face_idx)
        print('')