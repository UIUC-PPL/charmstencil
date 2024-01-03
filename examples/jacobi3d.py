import numpy as np
import sys
import time
import cProfile
from charmstencil.linalg import norm
from charmstencil.stencil import Stencil, StencilNumpy
from charmstencil.interface import DebugInterface, CCSInterface

class Jacobi3D(Stencil):
    def __init__(self, n, interface):
        self.x, self.y = self.initialize(
            n, interface=interface, max_epochs=1000, odf=2,
            num_fields=2)
        #self.apply_boundary(100.)
        self.itercount = 0
        self.boundary_iter = True

    def iterate(self, nsteps):
        if self.boundary_iter:
            self.bc(100)
            self.boundary_iter = False
            return False
        self.exchange_ghosts(self.x)
        self.y[1:-1, 1:-1, 1:-1] = (1. / 6) * (self.x[:-2, 1:-1, 1:-1] + self.x[2:, 1:-1, 1:-1] +
                                               self.x[1:-1, :-2, 1:-1] + self.x[1:-1, 2:, 1:-1] +
                                               self.x[1:-1, 1:-1, :-2] + self.x[1:-1, 1:-1, 2:])
        self.x, self.y = self.y, self.x
        #self.x1, self.y1 = self.y1, self.x1
        self.itercount += 1
        if self.itercount % nsteps == 0:
            return False
        else:
            return True

    def bc(self, bc):
        self.x[0, :, :] = self.y[0, :, :] = bc
        self.x[-1, :, :] = self.y[-1, :, :] = bc
        self.x[:, 0, :] = self.y[:, 0, :] = bc
        self.x[:, -1, :] = self.y[:, -1, :] = bc
        self.x[:, :, 0] = self.y[:, :, 0] = bc
        self.x[:, :, -1] = self.y[:, :, -1] = bc

if __name__ == '__main__':
    interface = DebugInterface()
    #pr = cProfile.Profile()
    #interface = CCSInterface(sys.argv[1], 1234)
    #interface = None
    grid = Jacobi3D((128, 128, 128), interface)
    #pr.enable()
    grid.solve(5)
    #print("Test")
    grid.sync()
    print("Warm up done")
    start = time.time()
    grid.solve(1000)
    grid.sync()
    #pr.disable()
    #pr.dump_stats("profile.prof")
    print("Total time =", time.time() - start)
