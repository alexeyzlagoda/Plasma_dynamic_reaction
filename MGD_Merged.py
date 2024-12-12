from Config import *
import numpy as np
from scipy.integrate import solve_ivp
from Solver1 import *
from tqdm import tqdm
from Struct import *

# Граничные условия (замкнутые стенки)
def apply_boundary_conditions(rho, V, B, p):
    for var in [rho, V, B, p]:
        var[0, :] = var[-1, :] = 0
        var[:, 0] = var[:, -1] = 0
    return rho, V, B, p

# Производные по пространству
def grad_x(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def grad_y(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

def laplacian(f):
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 + \
           (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2

def continuity(rho, V):
    drho_dt = - (grad_x(rho * V) + grad_y(rho * V))
    return drho_dt

# Уравнение движения
def momentum(rho, V, p, B):
    dV_dt = - V * grad_x(V) - V * grad_y(V) \
            - grad_x(p) / rho + nu * laplacian(V)
    return dV_dt

# Уравнение индукции
def induction(B, V):
    dB_dt = grad_x(V * B) - grad_y(V * B) + eta * laplacian(B)
    return dB_dt

# Уравнение энергии
def energy(rho, V, p, B):
    e = p / (gamma - 1) + 0.5 * rho * V**2 + 0.5 * B**2 / mu
    de_dt = -grad_x((e + p) * V) - grad_y((e + p) * V)
    return de_dt

def F(t,IncData):
    rho, V, p, B, T, n_D,  n_T , n_He = IncData.reshape((8,Nx, Ny))[0],IncData.reshape((8,Nx, Ny))[1], IncData.reshape((8,Nx, Ny))[2], IncData.reshape((8,Nx, Ny))[3], IncData.reshape((8,Nx, Ny))[4], IncData.reshape((8,Nx, Ny))[5], IncData.reshape((8,Nx, Ny))[6], IncData.reshape((8,Nx, Ny))[7]
    dT, dn_D,  dn_T , dn_He  = UpdateTemp(T, n_D,  n_T , n_He)
    dE = energy(rho, V, p, B) + 3 * k_B * T
    dB = induction(B, V)
    dV = momentum(rho, V, p, B)
    drho = continuity(rho, V)
    return np.concatenate([drho.ravel(), dV.ravel(), np.zeros((Nx, Ny)).ravel(), dB.ravel(), dT.ravel(), dn_D.ravel(),  dn_T.ravel() , dn_He.ravel()])

def solveMe(Duration, Data:SolverObj):    
    sol = solve_ivp(F,(0, Duration),Data)
    return sol