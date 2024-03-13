import open3d as o3d
import numpy as np
import time
from graphics.colormap import ColorMap

class Visualizer:

    def __init__(self, container=[10,10,10], voxel_size=0.05):
        '''
        Create a visualizer object that can visualize the packing space.

        Parameters
        ----------
            `container` : 3-dimensional integer vector, optional
                the dimensions of the container.
                format: `[width, height, depth]`
            `voxel_size` : float, optional
                the mesh size of each voxel.
        '''

        # set the container dimensions
        self.width = container[0]
        self.height = container[1]
        self.depth = container[2]
        self.voxel_size = voxel_size
        self.started = False

        # create a color map
        self.c = ColorMap()

        # line set
        points = [
            [0, 0, 0],
            [self.width * voxel_size, 0, 0],
            [0, self.height * voxel_size, 0],
            [self.width * voxel_size, self.height * voxel_size, 0],
            [0, 0, self.depth * voxel_size],
            [self.width * voxel_size, 0, self.depth * voxel_size],
            [0, self.height * voxel_size, self.depth * voxel_size],
            [self.width * voxel_size, self.height * voxel_size, self.depth * voxel_size]
        ]
        lines = [
            [0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7]
        ]
        self.line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(lines))])

    def start(self, title='Online 3D Irregular Packing', width=1024, height=768):
        '''
        Start the visualizer.

        Parameters
        ----------
            `title` : str, optional
                the title of the window.
            `width` : int, optional
                the width of the window.
            `height` : int, optional
                the height of the window.
        '''

        # initialize the visualizer
        o3d.visualization.gui.Application.instance.initialize()
        self.w = o3d.visualization.O3DVisualizer(title, width, height)
        self.w.set_background((1, 1, 1, 1), None)

        # add initial geometries
        self.w.add_geometry('voxel_grid', self.line_set, is_visible=False)
        self.w.add_geometry('container', self.line_set)

        # reset camera
        self.w.reset_camera_to_default()

        # start rendering
        self.started = True
        o3d.visualization.gui.Application.instance.add_window(self.w)
        o3d.visualization.gui.Application.instance.run()

    def await_start(self, timeout=10):
        '''
        Wait until the visualizer is started.

        Parameters
        ----------
            `timeout` : float, optional
                the maximum time to wait for the visualizer to start (default: 10 seconds).
        '''

        start_time = time.time()
        while not self.started:
            if time.time() - start_time > timeout:
                raise TimeoutError('visualizer did not start.')

    def update(self, container):
        '''
        Update the visualizer with the new container.
        All non-zero elements are considered as packed and will be drawn in different colors.

        Parameters
        ----------
            `container` : 3-dimensional matrix
                the packing space.
        '''

        # verify container dimensions
        assert container.shape[0] == self.width, f'{container.shape[0]} != {self.width}'
        assert container.shape[1] == self.height, f'{container.shape[1]} != {self.height}'
        assert container.shape[2] == self.depth, f'{container.shape[2]} != {self.depth}'

        # assert gui is ready
        assert self.started, 'visualizer was not started.'

        # reshape the array into (N, 3) form (only selects elements that are not 0)
        points = np.argwhere(container)

        # create a point cloud from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points * self.voxel_size)
        pcd.translate([self.voxel_size/2, self.voxel_size/2, self.voxel_size/2]) # center w.r.t. line set

        # color all voxels
        colors = [self.c.get_color(container[i, j, k]) for i, j, k in points]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # create a voxel grid from the point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        
        # update the visualizer
        self.w.remove_geometry('voxel_grid')
        self.w.add_geometry('voxel_grid', voxel_grid)
        self.w.post_redraw()
