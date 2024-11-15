import torch
import kaolin as kal
from tqdm import tqdm
from torch_scatter import scatter_max

def get_camera_from_views(absolute_elevations, relative_azimuths, r):
    # Convert elevation angle to polar angle
    polars_angle_deg = 90 - absolute_elevations
    polars_from_Y = torch.deg2rad(polars_angle_deg)

    thetas_from_X_deg = relative_azimuths
    thetas_from_Z_deg = thetas_from_X_deg + 90
    thetas_from_Z = torch.deg2rad(thetas_from_Z_deg)

    ys = r * torch.cos(polars_from_Y)
    # Projection of r onto the world ZX plane (without considering azimuth angle) => project r onto the Z axis
    zs = r * torch.sin(polars_from_Y) * torch.cos(thetas_from_Z)
    xs = r * torch.sin(polars_from_Y) * torch.sin(thetas_from_Z)

    # The vector (x_w,y_w,z_w) is the initial vector which will be rotated about the vertical axis Y_o of the object frame
    # by the relative azimuth angle, theta_from_Z_o.

    positions = torch.stack([xs, ys, zs], dim=1)
    look_at = torch.zeros_like(positions)  #MJ: Look at the root of the world frame

    # JA: As the camera's up direction is along the positive Y-axis in Kaolin, camera_up_direction is set to
    # [0, 1, 0] to match this convention
    camera_up_direction = torch.tensor([0.0, 1.0, 0.0], device=positions.device).unsqueeze(0).repeat(positions.size(0), 1)
    camera_transforms = kal.render.camera.generate_transformation_matrix(positions, look_at, camera_up_direction)

    return camera_transforms

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
