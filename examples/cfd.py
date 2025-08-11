from charmstencil.kernel import plot_kernel_graphs
from charmstencil.array import create_array
from charmstencil.dag import show_dag, disable_fusion, get_active_dag, set_max_depth
from charmstencil.interface import CCSInterface, set_interface
import numpy as np
import sys
import time

#disable_fusion()

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[2:, 1:-1] - u[0:-2, 1:-1]) / 
                     (2 * dx) + (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy)) -
                    ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx))**2 -
                      2 * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dy) *
                           (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dx))-
                          ((v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dy))**2))


def pressure_poisson(p, pn, dx, dy, b):    
    for q in range(nit):
        pn[1:-1, 1:-1] = (((p[2:, 1:-1] + p[0:-2, 1:-1]) * dy**2 + 
                          (p[1:-1, 2:] + p[1:-1, 0:-2]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])
        
        pn[-1, :] = pn[-2, :] # dp/dx = 0 at x = 2
        pn[:, 0] = pn[:, 1]   # dp/dy = 0 at y = 0
        pn[0, :] = pn[1, :]   # dp/dx = 0 at x = 0
        pn[:, -1] = pn[:, -2]    # p = 0 at y = 2
        #pn[0, 0] = 0

        p, pn = pn, p

    return p
        

def cavity_flow(nt, u, v, un, vn, dt, dx, dy, p, pn, rho, nu, b):    
    for n in range(0,nt):        
        build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, pn, dx, dy, b)
        
        un[1:-1, 1:-1] = (u[1:-1, 1:-1]-
                         u[1:-1, 1:-1] * dt / dx *
                        (u[1:-1, 1:-1] - u[0:-2, 1:-1]) -
                         v[1:-1, 1:-1] * dt / dy *
                        (u[1:-1, 1:-1] - u[1:-1, 0:-2]) -
                         dt / (2 * rho * dx) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 *
                        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]) +
                         dt / dy**2 *
                        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])))

        vn[1:-1,1:-1] = (v[1:-1, 1:-1] -
                        u[1:-1, 1:-1] * dt / dx *
                       (v[1:-1, 1:-1] - v[0:-2, 1:-1]) -
                        v[1:-1, 1:-1] * dt / dy *
                       (v[1:-1, 1:-1] - v[1:-1, 0:-2]) -
                        dt / (2 * rho * dy) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 *
                       (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) +
                        dt / dy**2 *
                       (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2])))

        un[:, 0]  = 0
        un[0, :]  = 0
        un[-1, :] = 0
        un[:, -1] = 1    # Set velocity on cavity lid equal to 1

        vn[:, 0]  = 0
        vn[:, -1] = 0
        vn[0, :]  = 0
        vn[-1, :] = 0

        u, un = un, u
        v, vn = vn, v

    return u, v, p

set_max_depth(200)

n = int(sys.argv[1])

u1 = create_array((n, n))
u2 = create_array((n, n))

v1 = create_array((n, n))
v2 = create_array((n, n))

p1 = create_array((n, n))
p2 = create_array((n, n))

b = create_array((n, n))

c = 1.
dx = 2. / (n - 1)
dy = 2. / (n - 1)

rho = 1
nu = .1
dt = .001

nt = 1000
nit = 30

interface = CCSInterface('192.168.1.114', 1234, odf=4)
set_interface(interface)

u, v, p = cavity_flow(1, u1, v1, u2, v2, dt, dx, dy, p1, p2, rho, nu, b)

interface.sync()
#show_dag()
#get_active_dag().clear()

start = time.time()
u, v, p = cavity_flow(30, u1, v1, u2, v2, dt, dx, dy, p1, p2, rho, nu, b)
interface.sync()

print(f"Execution took {time.time() - start} seconds")

#show_dag()
#plot_kernel_graphs()

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