import kaolin as kal
import torch
import numpy as np
from loguru import logger
class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest', fovyangle=np.pi / 3):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(fovyangle).to(device) # JA: This is the field of view
                                                                                        # It is currently set to 60deg.

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)

        pos = torch.tensor([x, y, z]).unsqueeze(0)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        camera_up_direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, camera_up_direction)
        return camera_proj
    
    @staticmethod
    def get_camera_from_multiple_view(elev, azim, r, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)

        pos = torch.stack([x, y, z], dim=1)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        camera_up_direction = torch.ones_like(pos) * torch.tensor([0.0, 1.0, 0.0]).to(pos.device)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, camera_up_direction)
        return camera_proj


    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, 'depth map should be negative' # JA: In the standard computer graphics, the camera view direction is the negative z axis
        object_mask = depth_map != 0 # JA: The region where the depth map is not 0 means that it is the object region
        # depth_map[object_mask] = (depth_map[object_mask] - depth_map[object_mask].min()) / (
        #             depth_map[object_mask].max() - depth_map[object_mask].min())
        # depth_map = depth_map ** 4
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map[depth_map == 1] = 0 # Background gets largest value, set to 0

        return depth_map

    def normalize_multiple_depth(self, depth_maps):
        assert (depth_maps.amax(dim=(1, 2)) <= 0).all(), 'depth map should be negative'
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

    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, radius=2, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)

    def render_multiple_view(self, mesh, face_attributes, elev, azim, radius, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(elev, azim, r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_multiple_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None):
        dims = self.dim if dims is None else dims

        if render_cache is None:

            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)
            # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
            depth_map = self.normalize_depth(depth_map)

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)
            uv_features = uv_features.detach()

        else:
            # logger.info('Using render cache')
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map']
        mask = (face_idx > -1).float()[..., None]

        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
                        # JA: Interpolates texture_maps by dense or sparse texture_coordinates (uv_features).
                        # This function supports sampling texture coordinates for:
                        # 1. An entire 2D image
                        # 2. A sparse point cloud of texture coordinates.

        image_features = image_features * mask # JA: image_features refers to the render image
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        normals_image = face_normals[0][face_idx, :] #MJ: what happens when face_idx[0,i,j] is -1 (non-face, background)?
        #MJ: an index of -1 refers to the last element in the array, but in this context, it is meant to indicate an invalid index.



        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map}

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache

    def render_multiple_view_texture(self, verts, faces, elev, azim, radius,
                                   look_at_height=0.0, dims=None):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_multiple_view(
            elev, azim, r=radius,
            look_at_height=look_at_height
        )
        # JA: Since the function prepare_vertices is specifically designed to move and project vertices to camera space and then index them with faces, the face normals returned by this function are also relative to the camera coordinate system. This follows from the context provided by the documentation, where the operations involve transforming vertices into camera coordinates, suggesting that the normals are computed in relation to these transformed vertices and thus would also be in camera coordinates.
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts, faces, self.camera_projection, camera_transform=camera_transform)
        # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
        depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                            face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
        depth_map = self.normalize_multiple_depth(depth_map)

        return depth_map.permute(0, 3, 1, 2)

    def project_uv_single_view(self, verts, faces, uv_face_attr, elev=0, azim=0, radius=2,
                               look_at_height=0.0, dims=None, background_type='none'):
        # project the vertices and interpolate the uv coordinates

        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                          face_vertices_image, uv_face_attr)
        return face_vertices_image, face_vertices_camera, uv_features, face_idx

    def project_single_view(self, verts, faces, elev=0, azim=0, radius=2,
                               look_at_height=0.0):
        # only project the vertices
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        return face_vertices_image
