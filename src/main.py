from graphics import visualizer
from shapes import random_polyomino, POLYOMINOES, ADJUSTED_PROB
import numpy as np
from threading import Thread

if __name__ == '__main__':
    vis = visualizer.Visualizer()

    # create random shape
    A = random_polyomino(POLYOMINOES, ADJUSTED_PROB)
    container = np.zeros((10, 10, 10))
    container[:A.shape[0],:A.shape[1],:A.shape[2]] = A

    # add the container to the visualizer
    def add_container():
        vis.await_start()
        vis.update(container)
    Thread(target=add_container).start()

    vis.start()
