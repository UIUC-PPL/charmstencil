from charmstencil.kernel import plot_kernel_graphs
from charmstencil.array import create_array
from charmstencil.dag import show_dag, get_active_dag, disable_fusion
from charmstencil.interface import CCSInterface
import numpy as np
import sys
from time import sleep

disable_fusion()

interface = CCSInterface('192.168.1.114', 1234, odf=2)

n = int(sys.argv[1])

u1 = create_array((n, n))
#u2 = create_array((n, n))

interface.rescale(2)
sleep(2)  # wait for rescale to take effect

u1[0, :] = 1

interface.rescale(2)
sleep(2)

u1[0, :] = 1
#boundary(u1)
#boundary(u2)

#for i in range(2):
#    u2[1:-1, 1:-1] = 0.25 * (u1[:-2, 1:-1] + u1[2:, 1:-1] + u1[1:-1, :-2] + u1[1:-1, 2:])
#    u1, u2 = u2, u1

interface.sync()
