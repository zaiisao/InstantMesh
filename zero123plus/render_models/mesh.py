import kaolin as kal
import torch

import copy

class Mesh:
    def __init__(self,obj_path, device):
        # from https://github.com/threedle/text2mesh

        if ".obj" in obj_path:
            try:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=True, heterogeneous_mesh_handler=kal.io.utils.heterogeneous_mesh_handler_naive_homogenize)
            except:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=False, heterogeneous_mesh_handler=kal.io.utils.heterogeneous_mesh_handler_naive_homogenize)

        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.normals, self.face_area = self.calculate_face_normals(self.vertices, self.faces)
        self.ft = mesh.face_uvs_idx
        self.vt = mesh.uvs

    @staticmethod
    def calculate_face_normals(vertices: torch.Tensor, faces: torch.Tensor):
        """
        calculate per face normals from vertices and faces
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=-1)
        twice_area = torch.norm(n, dim=-1)
        n = n / twice_area[:, None]
        return n, twice_area / 2

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.std(torch.norm(verts, p=2, dim=1))
        verts /= scale
        mesh.vertices = verts
        return mesh

    def normalize_mesh(self,inplace=False, target_scale=1, dy=0):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts = verts - center # JA: reposition vertices with respect to the center
        scale = torch.max(torch.norm(verts, p=2, dim=1)) # JA: scale is the length of the furthest vertex from the center
        verts = verts / scale # JA: normalize vertices relative to the largest coordinates. The normalized vertices range from [-1, 1]
        verts *= target_scale # JA: x, y, z: [-1, 1] -> [-0.6, 0.6]
        verts[:, 1] += dy # JA: y: [-0.6, 0.6] -> [-0.35, 0.85]
        mesh.vertices = verts
        return mesh

