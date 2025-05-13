from charmstencil.kernel import kernel, plot_kernel_graphs
from charmstencil.array import create_array
from charmstencil.dag import show_dag
from charmstencil.interface import CCSInterface
import numpy as np
import sys


# @kernel
# def boundary1(u1, u2):
#     u1[0, :] = 1
#     u2[0, :] = 1

# @kernel
# def boundary2(u1, u2):
#     u1[-1, :] = 1
#     u2[-1, :] = 1

# @kernel
# def boundary3(u1, u2):
#     u1[:, 0] = 1
#     u2[:, 0] = 1

# @kernel
# def boundary4(u1, u2):
#     u1[:, -1] = 1
#     u2[:, -1] = 1

@kernel
def init1(u, v):
    u[:,:] = 1
    v[:,:] = 1

@kernel
def init2(u, v, dx, dy):
    u[int(0.5/dx):int(1/dx+1),int(0.5/dy):int(1/dy+1)] = 2
    v[int(0.5/dx):int(1/dx+1),int(0.5/dy):int(1/dy+1)] = 2

@kernel
def burgers(u1, u2, v1, v2, nu, dt, dx, dy):
    u2[1:-1,1:-1] = (u1[1:-1,1:-1] - 
                    dt / dx * u1[1:-1,1:-1] * (u1[1:-1,1:-1] - u1[0:-2,1:-1]) -
                    dt / dy * v1[1:-1,1:-1] * (u1[1:-1,1:-1] - u1[1:-1,0:-2]) +
                   nu * dt / dx**2 * (u1[2:,1:-1] - 2 * u1[1:-1,1:-1] + u1[:-2,1:-1]) +
                   nu * dt / dy**2 * (u1[1:-1,2:] - 2 * u1[1:-1,1:-1] + u1[1:-1,:-2]))
    v2[1:-1,1:-1] = (v1[1:-1,1:-1] - 
                    dt / dx * u1[1:-1,1:-1] * (v1[1:-1,1:-1] - v1[0:-2,1:-1]) -
                    dt / dy * v1[1:-1,1:-1] * (v1[1:-1,1:-1] - v1[1:-1,0:-2]) +
                   nu * dt / dx**2 * (v1[2:,1:-1] - 2 * v1[1:-1,1:-1] + v1[:-2,1:-1]) +
                   nu * dt / dy**2 * (v1[1:-1,2:] - 2 * v1[1:-1,1:-1] + v1[1:-1,:-2]))

n = int(sys.argv[1])

u1 = create_array((n, n))
u2 = create_array((n, n))
v1 = create_array((n, n))
v2 = create_array((n, n))

c = 1.
dx = 2. / (n - 1)
dy = 2. / (n - 1)
nu = 0.01
sigma = 0.2
dt = sigma * dx

interface = CCSInterface('10.193.151.206', 1234, odf=1)

init1(u1, v1)
init1(u2, v2)
init2(u1, v1, dx, dy)

for i in range(10):
  burgers(u1, u2, v1, v2, nu, dt, dx, dy)
  u1, u2 = u2, u1
  v1, v2 = v2, v1

interface.execute()

for i in range(100):
  burgers(u1, u2, v1, v2, nu, dt, dx, dy)
  u1, u2 = u2, u1
  v1, v2 = v2, v1

interface.execute()

#plot_kernel_graphs()
#show_dag()

# uhost = u1.get(interface)
# vhost = v1.get(interface)
# print(uhost)
# import matplotlib.pyplot as plt
# from matplotlib import cm

# x = np.linspace(0, 2, n)
# y = np.linspace(0, 2, n)

# uhost = np.array(uhost).reshape(u1.shape)
# vhost = np.array(vhost).reshape(v1.shape)

# fig = plt.figure(figsize=(11, 7), dpi=100)
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(x, y)
# ax.plot_surface(X, Y, uhost[:], cmap=cm.viridis, rstride=1, cstride=1)
# ax.plot_surface(X, Y, vhost[:], cmap=cm.viridis, rstride=1, cstride=1)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# plt.show()