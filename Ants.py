import math
import time
import numpy
import pandas
import random
import matplotlib
import numpy.random as nrand
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize


class AntColonyOptimization:
    def __init__(self):
        pass


class Grid:
    def __init__(self, height, width):
        self.dim = numpy.array([height, width])
        self.grid = numpy.empty((height, width), dtype=Datum)
        self.rand_grid(0.15)

        plt.ion()
        plt.figure(figsize=(10, 10))
        self.plot_grid("images_three/" + "Init.png")

    def rand_grid(self, sparse):
        self.grid = numpy.empty((self.dim[0], self.dim[1]), dtype=Datum)
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if random.random() <= sparse:
                    if random.randint(0, 1) == 0:
                        self.grid[y][x] = Datum(nrand.normal(1, 0.5, 10))
                    else:
                        self.grid[y][x] = Datum(nrand.normal(5, 0.5, 10))

    def matrix_grid(self):
        ll = []
        for y in range(self.dim[0]):
            l = []
            for x in range(self.dim[1]):
                if self.grid[y][x] is not None:
                    # x_loc, y_loc, n, c, grid, dim
                    l.append(self.grid[y][x].density(x, y, 1, 5, self.get_grid(), self.dim))
                else:
                    l.append(-1)
            ll.append(l)
        m = numpy.matrix(ll)
        return m

    def plot_grid(self, name):
        # Cool colour map found here - http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        plt.matshow(self.matrix_grid(), cmap="Greens", fignum=0)
        plt.savefig(name + '.png')
        plt.draw()

    def get_grid(self):
        return self.grid


class Ant:
    def __init__(self, y, x, grid):
        self.loc = numpy.array([y, x])
        self.carrying = grid.get_grid()[y][x]
        self.grid = grid

    def move(self, n, c):
        """
        A recursive function for making ants move around the grid
        :param step_size: the size of each step
        """
        step_size = random.randint(1, 3)
        # Add some vector (-1,+1) * step_size to the ants location
        self.loc += nrand.randint(-1 * step_size, 1 * step_size, 2)
        # Mod the new location by the grid size to prevent overflow
        self.loc = numpy.mod(self.loc, self.grid.dim)
        # Get the object at that location on the grid
        o = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        # If the cell is occupied, move again
        if o is not None:
            # If the ant is not carrying an object
            if self.carrying is None:
                # Check if the ant picks up the object
                if self.p_pick_up(n, c) >= random.random():
                    # Pick up the object and rem from grid
                    self.carrying = o
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = None
                # If not then move
                else:
                    self.move(n, c)
            # If carrying an object then just move
            else:
                self.move(n, c)
        # If on an empty cell
        else:
            if self.carrying is not None:
                # Check if the ant drops the object
                if self.p_drop(n, c) >= random.random():
                    # Drop the object at the empty location
                    self.grid.get_grid()[self.loc[0]][self.loc[1]] = self.carrying
                    self.carrying = None

    def p_pick_up(self, n, c):
        """
        Returns the probability of picking up an object
        :param n: the neighbourhood size
        :return: probability of picking up
        """
        ant = self.grid.get_grid()[self.loc[0]][self.loc[1]]
        return 1 - self.density(ant, n, c)

    def p_drop(self, n, c):
        """
        Returns the probability of dropping an object
        :return: probability of dropping
        """
        ant = self.carrying
        return self.density(ant, n, c)

    def density(self, dat, n, c):
        x = self.loc[0] - n
        y = self.loc[1] - n
        g = self.grid.get_grid()
        d = self.grid.dim
        return dat.density(x, y, n, c, g, d)


class Data:
    def __init__(self):
        pass


class Datum:
    def __init__(self, data):
        self.data = data

    def similarity(self, datum):
        diff = numpy.abs(self.data - datum.data)
        return numpy.sum(diff**2)

    def condense(self):
        return numpy.sum(self.data)

    def bias(self):
        return numpy.prod(self.data + 1) % 2

    def density(self, x_loc, y_loc, n, c, grid, dim):
        x = x_loc - n
        y = y_loc - n
        total = 0.0
        for i in range((n*2)+1):
            xi = (x + i) % dim[0]
            for j in range((n*2)+1):
                if j != x_loc and i != y_loc:
                    yj = (y + j) % dim[1]
                    o = grid[xi][yj]
                    if o is not None:
                        s = self.similarity(o)
                        total += s
        density = total / (40 * (math.pow((n*2)+1, 2) - 1))
        density = max(min(density, 1), 0)
        t = math.exp(-c * density)
        probability = (1-t)/(1+t)
        return probability


def main():
    grid = Grid(50, 50)
    ants = []
    for i in range(10):
        ant = Ant(random.randint(0, 49), random.randint(0, 49), grid)
        ants.append(ant)
    for i in range(100000):
        for ant in ants:
            ant.move(random.randint(1, 2), 15)
        if i % 500 == 0:
            print(i)
            s = str(i).zfill(6)
            grid.plot_grid("images_three/" + s)
        if i == 100000:
            grid.plot_grid("images_three/end")


if __name__ == '__main__':
    main()