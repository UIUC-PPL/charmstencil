import numpy as np
from charmstencil.stencil import Stencil
from charmstencil.linalg import norm
from charmstencil.interface import DebugInterface, DummyInterface, CCSInterface

class Grid(Stencil):
    def __init__(self, n, interface):
        self.initialize(n, interface=interface, max_epochs=10)
        self.x = self.create_field(ghost_depth=1)
        self.y = self.create_field(ghost_depth=1)
        #self.apply_boundary(100.)
        self.threshold = 1e-8
        self.itercount = 0

    def boundary(self, bc):
        self.x[0] = self.y[0] = bc
        self.x[-1] = self.y[-1] = bc

    def iterate(self, nsteps):
        self.exchange_ghosts(self.x)
        self.y[1:-1] = 0.5 * (self.x[:-2] + self.x[2:])
        self.x, self.y = self.y, self.x
        self.itercount += 1
        if self.itercount % nsteps == 0:
            #return norm(self.x - self.y, np.inf).get() > self.threshold
            return False
        else:
            return True

if __name__ == '__main__':
    interface = DebugInterface()
    #interface = DummyInterface()
    #interface = CCSInterface("172.17.0.1", 10000)
    grid = Grid(100, interface)
    grid.solve(9)
