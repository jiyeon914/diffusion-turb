import math
pi = math.pi

import numpy as np
from numpy.fft import fftn as fftn_cpu
from numpy.fft import ifftn as ifftn_cpu
import cupy as cp
from cupy.fft import fftn, ifftn

### Statistics ###
def divergence_free(uk, vk, wk, KX, KY, KZ):
    continuityk = KX*uk + KY*vk + KZ*wk
    continuity = ifftn(continuityk, axes=(-3,-2,-1), norm='forward').real
    return continuity

def e_spectrum(uk, vk, wk, k_round, k_sq, k_index):
    Eu_3D, Ev_3D, Ew_3D = 2*pi*k_sq*uk*cp.conjugate(uk), 2*pi*k_sq*vk*cp.conjugate(vk), 2*pi*k_sq*wk*cp.conjugate(wk)
    E_3D = (Eu_3D + Ev_3D + Ew_3D).real
    spectrum = cp.zeros_like(k_index, dtype=np.float64)
    for i in k_index:
        spectrum[i] = cp.sum(E_3D[i == k_round])
    return spectrum

def dissipation(uk, vk, wk, KX, KY, KZ, nu): # 2*nu*<S_{ij}S_{ij}>
    u_xk, u_yk, u_zk = 1J*KX*uk, 1J*KY*uk, 1J*KZ*uk
    v_xk, v_yk, v_zk = 1J*KX*vk, 1J*KY*vk, 1J*KZ*vk
    w_xk, w_yk, w_zk = 1J*KX*wk, 1J*KY*wk, 1J*KZ*wk
    u_x, u_y, u_z = ifftn(u_xk, axes=(-3,-2,-1), norm='forward').real, ifftn(u_yk, axes=(-3,-2,-1), norm='forward').real, ifftn(u_zk, axes=(-3,-2,-1), norm='forward').real
    v_x, v_y, v_z = ifftn(v_xk, axes=(-3,-2,-1), norm='forward').real, ifftn(v_yk, axes=(-3,-2,-1), norm='forward').real, ifftn(v_zk, axes=(-3,-2,-1), norm='forward').real
    w_x, w_y, w_z = ifftn(w_xk, axes=(-3,-2,-1), norm='forward').real, ifftn(w_yk, axes=(-3,-2,-1), norm='forward').real, ifftn(w_zk, axes=(-3,-2,-1), norm='forward').real

    d = 2*nu*cp.mean(u_x**2 + v_y**2 + w_z**2 + 2*((0.5*(u_y + v_x))**2 + (0.5*(v_z + w_y))**2 + (0.5*(w_x + u_z))**2))
    return d

def rms_vel_mag(u, v, w):
    u_rms, v_rms, w_rms = cp.std(u, axis=(-3,-2,-1)), cp.std(v, axis=(-3,-2,-1)), cp.std(w, axis=(-3,-2,-1))
    return cp.sqrt((u_rms**2 + v_rms**2 + w_rms**2)/3)

def taylor_micro_scale(U_rms, d, nu):
    return cp.sqrt(15*nu*U_rms**2/d)

def reynolds_num(U_rms, lambda_g, nu):
    return U_rms*lambda_g/nu

def kolmogorov_length_scale(d, nu):
    return (nu**3/d)**(1/4)

def kolmogorov_time_scale(d, nu):
    return cp.sqrt(nu/d)

def cfl_number(u, v, w, dx, dy, dz, dt):
    CFL_cell = dt*(u/dx + v/dy + w/dz)
    return cp.max(CFL_cell)
