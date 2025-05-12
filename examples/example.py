from charmstencil.kernel import kernel, plot_kernel_graphs
from charmstencil.array import create_array
from charmstencil.dag import show_dag
from charmstencil.interface import CCSInterface
import numpy as np

#@kernel
#def boundary0(u1):
#    u1[:, :] = 0

@kernel
def boundary1(u1, u2):
    u1[0, :] = 1
    u2[0, :] = 1

@kernel
def boundary2(u1, u2):
    u1[-1, :] = 1
    u2[-1, :] = 1

@kernel
def boundary3(u1, u2):
    u1[:, 0] = 1
    u2[:, 0] = 1

@kernel
def boundary4(u1, u2):
    u1[:, -1] = 1
    u2[:, -1] = 1


@kernel
def jacobi(u1, u2):
    u2[1:-1, 1:-1] = 0.25 * (u1[:-2, 1:-1] + u1[2:, 1:-1] + u1[1:-1, :-2] + u1[1:-1, 2:])

u1 = create_array((16384, 16384))
u2 = create_array((16384, 16384))

#boundary0(u1)
boundary1(u1, u2)
boundary2(u1, u2)
boundary3(u1, u2)
boundary4(u1, u2)
for i in range(500):
    jacobi(u1, u2)
    u1, u2 = u2, u1

interface = CCSInterface('10.193.151.206', 1234, odf=4)
interface.execute()

# uhost = u1.get(interface)
# print(uhost)
# import matplotlib.pyplot as plt

# uhost = np.array(uhost).reshape(u1.shape)
# plt.imshow(np.array(uhost), cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("Heatmap of u1")
# plt.clim(0, 1)
# plt.show()
#show_dag()