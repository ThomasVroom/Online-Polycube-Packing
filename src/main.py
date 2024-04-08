from graphics import visualizer
from shapes.shape_generator import ShapeGenerator
from shapes.container import Container
from threading import Thread

if __name__ == '__main__':

    c = Container(10, 10, 10)
    vis = visualizer.Visualizer(c.get_dimensions())
    g = ShapeGenerator(upper_bound=3)
    
    p1 = g.get_random_polycube()
    p = p1.get_rotations()
    for i, p0 in enumerate(p):
        c.add(p0, ((i * 3) % 9, int(i / 9) * 3, int(i / 3) * 3 % 9))
    
    def add_container():
        vis.await_start()
        vis.update(c)
    Thread(target=add_container).start()
    vis.start()
