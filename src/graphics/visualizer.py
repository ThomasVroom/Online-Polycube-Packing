from graphics.colormap import ColorMap
import open3d as o3d
import numpy as np
import time

class Visualizer:

    def __init__(self, dim, voxel_size=0.05):
        '''
        Create a visualizer object that can visualize the packing space.

        Parameters
        ----------
            `dim` : 3-dimensional tuple
                the dimensions of the container.
            `voxel_size` : float, optional
                the mesh size of each voxel.
        '''

        # set the container dimensions
        self.width = dim[0]
        self.height = dim[1]
        self.depth = dim[2]
        self.voxel_size = voxel_size
        self.started = False

        # create a color map
        self.c = ColorMap()

        # line set
        points = [
            [0, 0, 0], # unused because of axes
            [self.width * voxel_size, 0, 0],
            [0, self.height * voxel_size, 0],
            [self.width * voxel_size, self.height * voxel_size, 0],
            [0, 0, self.depth * voxel_size],
            [self.width * voxel_size, 0, self.depth * voxel_size],
            [0, self.height * voxel_size, self.depth * voxel_size],
            [self.width * voxel_size, self.height * voxel_size, self.depth * voxel_size]
        ]
        lines = [
            [1, 3],[2, 3],[4, 5],[4, 6],[5, 7],[6, 7],[1, 5],[2, 6],[3, 7]
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
        self.w.show_axes = True
        self.w.show_settings = False

        # labels
        self.labels = []
        self.labels_visible = True

        # register actions
        def toggle_labels(vis):
            if self.labels_visible: # labels are visible, hide them
                vis.clear_3d_labels()
                self.labels_visible = False
            else: # labels are hidden, show them
                for label, id in self.labels:
                    vis.add_3d_label(label, str(int(id)))
                self.labels_visible = True
        self.w.add_action('toggle labels', toggle_labels)

        # reset camera
        self.w.reset_camera_to_default()

        # start rendering
        o3d.visualization.gui.Application.instance.add_window(self.w)
        self.started = True # this might crash if await_start checks before the next line, but that margin is very small
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
        Update the visualizer based on a container.

        Parameters
        ----------
            `container` : `Container`
                the packing space.
        '''

        # assert gui is ready
        assert self.started, 'visualizer was not started.'

        # reshape the array into (N, 3) form (only selects elements that are not 0)
        points = np.argwhere(container.matrix)

        # create a point cloud from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points * self.voxel_size)
        pcd.translate([self.voxel_size/2, self.voxel_size/2, self.voxel_size/2]) # center w.r.t. line set

        # color all voxels
        colors = [self.c.get_color(container.matrix[i, j, k]) for i, j, k in points]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # create a voxel grid from the point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)

        # update labels
        for id in container.get_ids():
            points = np.argwhere(container.matrix == id)
            mean = np.mean(np.asarray(points), axis=0) * self.voxel_size
            self.labels.append((mean + [0.5 * self.voxel_size, 0, 0.5 * self.voxel_size], id))
        if self.labels_visible:
            self.w.clear_3d_labels()
            for label, id in self.labels:
                self.w.add_3d_label(label, str(int(id)))

        # update the visualizer
        self.w.remove_geometry('voxel_grid')
        self.w.add_geometry('voxel_grid', voxel_grid)
        self.w.post_redraw()
