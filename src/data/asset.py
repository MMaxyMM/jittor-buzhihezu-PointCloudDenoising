from dataclasses import dataclass

from numpy import ndarray
from typing import Dict, Optional

import numpy as np
import os

@dataclass
class Asset():
    path: Optional[str]=None # where is the asset loaded from
    
    cls: Optional[str]=None # cls
    
    vertices: Optional[ndarray]=None # shape (N, 3)
    
    faces: Optional[ndarray]=None # shape (F, 3)

    vertex_normals: Optional[ndarray]=None

    face_normals: Optional[ndarray]=None
    
    sampled_vertices: Optional[ndarray]=None

    # Normals paired one-to-one with sampled_vertices / cached_vertices.
    sampled_normals: Optional[ndarray]=None

    # Original OBJ vertices stored alongside a cached surface-point pool.
    # AugmentSample may preserve a random subset of these during training.
    cached_vertices: Optional[ndarray]=None

    cached_vertex_normals: Optional[ndarray]=None

    
    sampled_vertices_noisy: Optional[ndarray]=None
    
    meta: Optional[Dict]=None
    
    def transform(self, trans: ndarray):
        """trans: 4x4 affine matrix"""
        def _apply(v: ndarray, trans: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]
        
        if self.vertices is not None:
            self.vertices = _apply(self.vertices, trans)
        if self.sampled_vertices is not None:
            self.sampled_vertices = _apply(self.sampled_vertices, trans)
        if self.sampled_vertices_noisy is not None:
            self.sampled_vertices_noisy = _apply(self.sampled_vertices_noisy, trans)
        normal_matrix = np.linalg.inv(trans[:3, :3]).transpose()
        for name in (
            "vertex_normals",
            "face_normals",
            "sampled_normals",
        ):
            normals = getattr(self, name)
            if normals is not None:
                transformed = np.matmul(normals, normal_matrix.transpose())
                lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
                setattr(
                    self,
                    name,
                    transformed / np.maximum(lengths, 1e-12),
                )

class Exporter(): # a simple parser
    
    @classmethod
    def _safe_make_dir(cls, path: str):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    @classmethod
    def export_obj(cls, vertices, path: str, precision: int=6):
        lines = []
        for v in vertices:
            lines.append(f'v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n')
        cls._safe_make_dir(path)
        f = open(path, "w")
        f.writelines(lines)
        f.close()