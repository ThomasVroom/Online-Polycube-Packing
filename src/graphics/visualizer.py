from src.graphics import ColorMap
import src.graphics.text_colors as tc
from src.environment import PackingEnv
from src.agents import Agent
import open3d as o3d
import numpy as np
import time

class Visualizer:

    def __init__(self, environment: PackingEnv, voxel_size: float=0.05, agent: Agent=None):
        '''
        Create a visualizer object that can visualize the packing environment.

        Parameters
        ----------
            `environment` : `PackingEnv`
                the packing environment that should be visualized.
            `voxel_size` : float, optional
                the mesh size of each voxel.
            `agent` : `Agent`, optional
                optional agent reference for controlling the packing through the UI.
        '''

        # set the environment variables
        self.environment = environment
        self.width, self.height, self.depth = environment.dimensions
        self.voxel_size = voxel_size
        self.agent = agent
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

    def start(self, title: str='Online 3D Irregular Packing', width: int=1024, height: int=768, labels: bool=True):
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
            `labels` : bool, optional
                whether to show labels for the packed shapes (can be toggled).
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

        # action for toggling labels
        self.labels_visible = labels
        def toggle_labels(vis):
            if self.labels_visible: # labels are visible, hide them
                vis.clear_3d_labels()
                self.labels_visible = False
            else: # labels are hidden, show them
                for label, id in self.labels:
                    vis.add_3d_label(label, str(int(id)))
                self.labels_visible = True
        self.w.add_action('toggle labels', toggle_labels)

        # actions for controlling the packing
        if self.agent is not None:
            def next_shape(_):
                if not self.environment.is_terminal():
                    self.environment.step(self.agent.get_action(self.environment))
                    self.update()
                else:
                        print(f'{tc.CRED}error: environment is terminal{tc.CEND}')
            def reset_environment(_):
                self.environment.reset()
                self.update()
            self.w.add_action('reset environment', reset_environment)
            self.w.add_action('next shape', next_shape)

        # reset camera
        self.w.reset_camera_to_default()

        # start rendering
        o3d.visualization.gui.Application.instance.add_window(self.w)
        self.started = True # this might crash if await_start checks before the next line, but that margin is very small
        o3d.visualization.gui.Application.instance.run()

    def await_start(self, timeout: float=10):
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
                raise TimeoutError(f'{tc.CRED}visualizer did not start{tc.CEND}')
        time.sleep(0.1) # give some time for the visualizer to finish starting

    def update(self):
        '''
        Update the visualizer based on the current environment.
        '''

        # assert gui is ready
        assert self.started, f'{tc.CRED}visualizer was not started{tc.CEND}'

        # reshape the array into (N, 3) form (only selects elements that are not 0)
        points = np.argwhere(self.environment.container.matrix)

        # create a point cloud from the points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points * self.voxel_size)
        pcd.translate([self.voxel_size/2, self.voxel_size/2, self.voxel_size/2]) # center w.r.t. line set

        # color all voxels
        colors = [self.c.get_color(self.environment.container.matrix[i, j, k], self.environment.np_random) for i, j, k in points]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # create a voxel grid from the point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)

        # update labels
        self.labels = []
        for id in self.environment.container.get_ids():
            points = np.argwhere(self.environment.container.matrix == id)
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
        print(f'{tc.CYELLOW2}total fitted: {len(self.environment.container.get_ids())}{tc.CEND}')
