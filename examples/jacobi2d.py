import numpy as np
from charmstencil.linalg import norm
from charmstencil.stencil import Stencil
from charmstencil.interface import DebugInterface, CCSInterface

class Grid(Stencil):
    def __init__(self, n, interface):
        self.x, self.y = self.initialize(
            (n, n), interface=interface, max_epochs=1000, odf=2,
            num_fields=2)
        #self.apply_boundary(100.)
        self.itercount = 0

    def iterate(self, nsteps):
        self.exchange_ghosts(self.x)
        self.y[1:-1, 1:-1] = 0.25 * (self.x[:-2, 1:-1] + self.x[2:, 1:-1] +
                                     self.x[1:-1, :-2] + self.x[1:-1, 2:])
        self.x, self.y = self.y, self.x
        self.itercount += 1
        return self.itercount == nsteps

    def boundary(self, bc):
        self.x[0, :] = self.y[0, :] = bc
        self.x[-1, :] = self.y[-1, :] = bc
        self.x[:, 0] = self.y[:, 0] = bc
        self.x[:, -1] = self.y[:, -1] = bc

if __name__ == '__main__':
    #interface = DebugInterface()
    interface = DebugInterface()
    grid = Grid((128, 128), interface)
    grid.solve(10)
