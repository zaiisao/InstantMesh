import kaolin as kal
import os
import torch
import math
import numpy as np
import pygltflib
import torchvision
import struct
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm
import json

def get_camera_transforms_from_cam_positions(positions): #MJ: azim, polar: tensors of shape [B,1]
    look_at = torch.zeros_like(positions)  #MJ: Look at the root of the world frame

    # JA: As the camera's up direction is along the positive Y-axis in Kaolin, camera_up_direction is set to
    # [0, 1, 0] to match this convention
    camera_up_direction = torch.tensor([0.0, 1.0, 0.0], device=positions.device).unsqueeze(0).repeat(positions.size(0), 1)

    camera_transforms = kal.render.camera.generate_transformation_matrix(positions, look_at, camera_up_direction)
    return camera_transforms

def compute_cam_pos_from_spherical_coord(polars_from_Y, thetas_from_Z, r):
    # The world Y component of the vector
    ys = r * np.cos(polars_from_Y)
    # Projection of r onto the world ZX plane (without considering azimuth angle) => project r onto the Z axis
    rs_proj_onto_ZX = r * np.sin(polars_from_Y)
    zs = r * np.sin(polars_from_Y) * np.cos(thetas_from_Z)  #MJ: = 1.48
    xs = r * np.sin(polars_from_Y) * np.sin(thetas_from_Z)

    # The vector (x_w,y_w,z_w) is the initial vector which will be rotated about the vertical axis Y_o of the object frame
    # by the relative azimuth angle, theta_from_Z_o.

    positions = np.array(list(zip(xs, ys, zs)))  #MJ: zip() creates an iterable

    return positions

def get_zero123plus_angles():
    relative_azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)  #Increment by 60 degs starting from 30
    absolute_elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)

    return relative_azimuths, absolute_elevations

def load_mesh(image_path):
    mesh_vertices = torch.load(os.path.join(image_path, 'mesh_vertices.pt')).cuda()
    mesh_faces = torch.load(os.path.join(image_path, 'mesh_faces.pt')).cuda()
    mesh_uvs = torch.load(os.path.join(image_path, 'mesh_uvs.pt')).cuda()
    mesh_face_uvs_idx = torch.load(os.path.join(image_path, 'mesh_face_uvs_idx.pt')).cuda()

    return mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx

def generate_object_masks_kaolin(mesh_path):
    mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx = load_mesh(mesh_path)

    relative_azimuths, absolute_elevations = get_zero123plus_angles()

    assert relative_azimuths.shape[0] == absolute_elevations.shape[0]
    num_viewpoints = relative_azimuths.shape[0]

    with open(os.path.join(mesh_path, 'azimuth.bin'), 'rb') as file:
        binary_data = file.read(2) #MJ:  reads the first 2 bytes from the binary file azimuth.bin.
        cond_azimuth = struct.unpack('h', binary_data)[0] #MJ:  Interprets the 2 bytes as a signed integer;'h' stands for a signed 2-byte (16-bit) integer

        relative_azimuths += cond_azimuth  #MJ: Add cond_azimuth to each element of azimuths_o

    # Convert the averaged quaternion to a rotation matrix
    # r = 0.5 / np.tan(np.radians(30/2))
    r = (0.5 * (1 + np.tan(np.radians(30/2)))) / np.tan(np.radians(30/2))

    # Convert elevation angle to polar angle
    polars_angle_deg = 90 - absolute_elevations
    polars_from_Y = np.radians(polars_angle_deg)

    thetas_from_X_deg = relative_azimuths
    thetas_from_Z_deg = thetas_from_X_deg + 90
    thetas_from_Z = np.radians(thetas_from_Z_deg)

    #MJ; compute the camera position in the world frame corresponding to the absolute polar  angle, poloar_from_Y_w,
    # and the relative azimuth angle, theta_from_Z_o
    cam_positions = compute_cam_pos_from_spherical_coord(polars_from_Y, thetas_from_Z, r)

    #MJ: set the current camera location in the world frame
    ## Convert directly to a single tensor on the GPU
    cam_positions_tensor = torch.tensor(cam_positions, dtype=torch.float32, device='cuda')
    camera_transforms = get_camera_transforms_from_cam_positions(cam_positions_tensor)

    fovyangle = math.radians(30)

    camera_projection = kal.render.camera.generate_perspective_projection(fovyangle).cuda()

    face_vertices_camera_one_mesh, face_vertices_image_one_mesh, face_normals_one_mesh = \
        kal.render.mesh.prepare_vertices(
            mesh_vertices[None],
            mesh_faces,
            camera_projection,
            camera_transform=camera_transforms
        )

    face_attributes_one_mesh = kal.ops.mesh.index_vertices_by_faces(
        mesh_uvs.repeat(num_viewpoints, 1, 1),
        mesh_face_uvs_idx.long()
    ).detach() # JA: Face attributes include the vertices of each face and the UV coordinates of each face

    uv_features_one_mesh, face_idx_one_mesh = kal.render.mesh.rasterize(
        320, 320, # JA: Zero123++ assumes each image size is 320x320
        face_vertices_camera_one_mesh[:, :, :, -1],
        face_vertices_image_one_mesh,
        face_attributes_one_mesh
    )   # JA: uv_features_one_mesh.shape = (6, 320, 320, 2):
        # It defines the UV coordinates of the texture map, to be assigned to each pixel ij of the rasterized image.

    uv_features = uv_features_one_mesh.detach()
    object_mask_bhw = (face_idx_one_mesh > -1).float() * 255

    return object_mask_bhw

def create_object_masks_blender(input_folder):
    # Create output folder if it doesn't exist
    object_masks = []

    # Loop through all the images in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png") and filename in ["001.png", "002.png", "003.png", "004.png", "005.png", "006.png"]:
            input_path = os.path.join(input_folder, filename)

            # Open the image and convert to RGBA if not already
            img = Image.open(input_path).convert("RGBA")
            img_data = torchvision.transforms.functional.pil_to_tensor(img)

            # Extract alpha channel
            alpha_channel = img_data[3]

            # Create the object mask (white for objects, black for background)
            mask = torch.where(alpha_channel > 0, 255, 0)

            object_masks.append(mask)

    return torch.stack(object_masks)

def load_image(image_path):
    """Load a binary image and convert it to a numpy array."""
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

def pixel_similarity(img1, img2):
    """Calculate pixel-by-pixel similarity for identical-size images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for pixel comparison.")

    total_pixels = torch.numel(img1)
    differing_pixels = torch.sum(img1 != img2)
    similarity = (total_pixels - differing_pixels) / total_pixels
    return similarity, differing_pixels, total_pixels

def calculate_similarity(img1, img2):
    """Calculate both pixel and cross-correlation similarity."""
    # Pixel-by-pixel similarity
    pixel_sim = pixel_similarity(img1, img2)

    return pixel_sim

base_folder = "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-000"
flagged_meshes = []

base_folders = [
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-000",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-001",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-002",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-003",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-004",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-005",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-006",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-007",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-008",
    "/home/sogang/mnt/db_1/jaehoon/objaverse/zero123plus-dataset/000-009",
]

non_flagged_meshes = []
flagged_meshes = []

all_meshes = []
# Iterate through each base folder
for base_folder in base_folders:
    # JA: Get a list of all subfolders first for tqdm

    for root, dirs, files in os.walk(base_folder):
        all_meshes.extend([os.path.join(root, subfolder) for subfolder in dirs])

# JA: Iterate through the subfolders with tqdm progress bar
bar = tqdm(all_meshes, desc=f"Calculating similarity")
for full_path in bar:
    try:
        bar.set_description(f"Calculating similarity; {len(flagged_meshes)} meshes are flagged")

        subfolder = os.path.basename(full_path)
        kal_object_masks_bhw = generate_object_masks_kaolin(full_path)
        # kal_object_masks_hw1_list = [kal_object_masks_bhw1[i].permute(1, 2, 0).cpu().numpy() * 255 for i in range(len(kal_object_masks_bhw1))]

        blender_object_masks_bhw = create_object_masks_blender(full_path)
        blender_object_masks_bhw = blender_object_masks_bhw.to(kal_object_masks_bhw.device)

        similarity, differing_pixels, total_pixels = calculate_similarity(kal_object_masks_bhw, blender_object_masks_bhw)
        # print(f"{subfolder} | Similarity score: {similarity:.4f} | {differing_pixels} pixels out of {total_pixels} pixels differ")

        if similarity < 0.95:
            flagged_meshes.append(subfolder)
        else:
            non_flagged_meshes.append(subfolder)
    except KeyboardInterrupt:
        pass
    except:
        print(f"An error occurred for the comparison of {full_path}; skipping...")
        continue

print("Flagged meshes are as follows:")
print(flagged_meshes)

data = {"folders": non_flagged_meshes}

# Write the dictionary to a JSON file
with open("non_flagged_meshes.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"Non-flagged meshes saved as non_flagged_meshes.json")

