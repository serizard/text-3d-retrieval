import open3d as o3d
import numpy as np

class Renderer():
    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(width=img_w, height=img_h, visible=False)
        self.visualizer.get_render_option().background_color = np.array([1, 1, 1])
        

    def normalize_align_mesh(self, mesh):
        oriented_bbox = mesh.get_oriented_bounding_box()
        max_extent = max(oriented_bbox.extent)
        scale = 1.0 / max_extent
        mesh.scale(scale = scale, center = oriented_bbox.get_center())
        center = oriented_bbox.get_center()
        mesh.translate(-center)
        return mesh



    def render(self, obj_file):
        self.visualizer.clear_geometries()
    
        normalized_mesh = self.normalize_align_mesh(obj_file)
        self.visualizer.add_geometry(normalized_mesh)
        
        bbox = normalized_mesh.get_oriented_bounding_box()
        center = bbox.get_center()
        extent = bbox.extent
        print(center, extent)
        
        fx = fy = 2
        cx = self.img_w / 2
        cy = self.img_h / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_w, self.img_h, fx, fy, cx, cy)
        
        camera_distance = extent[0] * 2
        camera_position = center + np.array([0, 0, camera_distance])
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = camera_position
        extrinsic = np.linalg.inv(extrinsic)
        
        new_intrinsics = o3d.camera.PinholeCameraParameters()
        new_intrinsics.intrinsic = intrinsic
        new_intrinsics.extrinsic = extrinsic
        
        view_control = self.visualizer.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(new_intrinsics, True)
        
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
        image = self.visualizer.capture_screen_float_buffer(do_render=True)
        
        return np.asarray(image)
    
    def close(self):
        self.visualizer.destroy_window()