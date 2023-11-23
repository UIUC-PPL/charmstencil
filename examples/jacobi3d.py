import numpy as np
import time
import cProfile
from charmstencil.linalg import norm
from charmstencil.stencil import Stencil
from charmstencil.interface import DebugInterface, CCSInterface

class Grid(Stencil):
    def __init__(self, n, interface):
        self.x, self.y = self.initialize(
            n, interface=interface, max_epochs=1000, odf=2,
            num_fields=2)
        #self.apply_boundary(100.)
        self.threshold = 1e-8
        self.itercount = 0

    def iterate(self, nsteps):
        self.exchange_ghosts(self.x)
        self.y[1:-1, 1:-1, 1:-1] = (1. / 6) * (self.x[:-2, 1:-1, 1:-1] + self.x[2:, 1:-1, 1:-1] +
                                               self.x[1:-1, :-2, 1:-1] + self.x[1:-1, 2:, 1:-1] +
                                               self.x[1:-1, 1:-1, :-2] + self.x[1:-1, 1:-1, 2:])
        self.x, self.y = self.y, self.x
        self.itercount += 1
        if self.itercount % nsteps == 0:
            #return norm(self.x - self.y, np.inf).get() > self.threshold
            return False
        else:
            return True

    def boundary(self, bc):
        self.x[0, :] = self.y[0, :] = bc
        self.x[-1, :] = self.y[-1, :] = bc
        self.x[:, 0] = self.y[:, 0] = bc
        self.x[:, -1] = self.y[:, -1] = bc

if __name__ == '__main__':
    #interface = DebugInterface()
    #pr = cProfile.Profile()
    interface = CCSInterface("172.17.0.1", 10000)
    grid = Grid((512, 512, 512), interface)
    #pr.enable()
    start = time.time()
    grid.solve(10)
    grid.sync()
    #pr.disable()
    #pr.dump_stats("profile.prof")
    print("Total time =", time.time() - start)
