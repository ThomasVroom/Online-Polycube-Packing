from graphics import visualizer
from shapes.shape_generator import ShapeGenerator
from shapes.container import Container
from threading import Thread

if __name__ == '__main__':

    c = Container(10, 10, 10)
    vis = visualizer.Visualizer(c.get_dimensions())
    g = ShapeGenerator(upper_bound=5)
    
    p = g.get_random_polycube()
    if c.fit(p, (0, 0, 0)):
        c.add(p, (0, 0, 0))
    
    def add_container():
        vis.await_start()
        vis.update(c)
    Thread(target=add_container).start()
    vis.start()
