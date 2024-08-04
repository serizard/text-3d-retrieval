import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

''' Incomplete code'''

class Renderer():
    def __init__(self, img_h, img_w):
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(img_w, img_h)
        
        self.renderer.scene.set_background([1,1,1,1])
        self.renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0, 0, 0))

        self.img_h = img_h
        self.img_w = img_w

    def render(self, obj_file, obj_name, cam_param, T_gk):
        material = o3d.visualization.rendering.MaterialRecord() 

        self.renderer.scene.add_geometry(name=obj_name, geometry=obj_file, material=material)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, cam_param.fx, cam_param.fy, cam_param.cx, cam_param.cy)

        new_cam_param = o3d.camera.PinholeCameraParameters()
        new_cam_param.intrinsic = intrinsic
        new_cam_param.extrinsic = np.linalg.inv(T_gk)
        
        self.renderer.setup_camera(new_cam_param.intrinsic, new_cam_param.extrinsic)

        bbox = obj_file.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.renderer.scene.camera.look_at(center, center + np.array([0, 0, -1]), np.array([0, -1, 0]))

        
        rgb = self.renderer.render_to_image()

        return rgb

    def close(self):
        self.renderer.scene.clear_geometry()
