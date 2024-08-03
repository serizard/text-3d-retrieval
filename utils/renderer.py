import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Renderer():
    def __init__(self, img_h, img_w):
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(img_w, img_h)
        
        self.renderer.scene.set_background([1, 1, 1, 1])  # White background
        self.renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0, 0, 0))  # No shadows

        self.img_h = img_h
        self.img_w = img_w

    def render(self, obj_file, obj_name, cam_param, T_gk):
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        self.renderer.scene.add_geometry(name=obj_name, geometry=obj_file, material=material)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, cam_param.fx, cam_param.fy, cam_param.cx, cam_param.cy)

        new_cam_param = o3d.camera.PinholeCameraParameters()
        new_cam_param.intrinsic = intrinsic
        new_cam_param.extrinsic = np.linalg.inv(T_gk)
        
        self.renderer.setup_camera(new_cam_param.intrinsic, new_cam_param.extrinsic)
        
        rgb = self.renderer.render_to_image()

        return rgb

    def close(self):
        self.renderer.scene.clear_geometry()
