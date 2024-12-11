
from scipy.integrate import solve_ivp
import numpy as np
from Config import *
#Граничные условия (замкнутые стенки)
def apply_boundary_conditions(rho, V, B, p):
    # Применяем граничные условия
    rho[0, :], rho[-1, :], rho[:, 0], rho[:, -1] = 0, 0, 0, 0
    V[0, :], V[-1, :], V[:, 0], V[:, -1] = 0, 0, 0, 0
    B[0, :], B[-1, :], B[:, 0], B[:, -1] = 0, 0, 0, 0
    p[0, :], p[-1, :], p[:, 0], p[:, -1] = 0, 0, 0, 0

    return rho, V, B, p

# Производные по пространству
def grad_x(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def grad_y(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

def laplacian(f):
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 + \
           (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2

# Уравнение непрерывности (сохранение массы)
def continuity(t,Data):
    MData = np.array(Data).reshape(4, Nx, Ny)
    rho = MData[0]
    V = MData[1]
    p = MData[2]
    B = MData[3]
    drho_dt = - (grad_x(rho * V) + grad_y(rho * V))
    return np.array(drho_dt).ravel()

# Уравнение движения
def momentum(t, Data):
    MData = np.array(Data).reshape(4, Nx, Ny)
    rho = MData[0]
    V = MData[1]
    p = MData[2]
    B = MData[3]
    dV_dt = - V * grad_x(V) - V * grad_y(V) \
            - grad_x(p) / rho + nu * laplacian(V)
    return np.array(dV_dt).ravel()

# Уравнение индукции
def induction(t, Data):
    MData = np.array(Data).reshape(4, Nx, Ny)
    rho = MData[0]
    V = MData[1]
    p = MData[2]
    B = MData[3]
    dB_dt = grad_x(V * B) - grad_y(V * B) + eta * laplacian(B)
    return np.array(dB_dt).ravel()

# Уравнение энергии
def energy(t, Data):
    MData = np.array(Data).reshape(4, Nx, Ny)
    rho = MData[0]
    V = MData[1]
    p = MData[2]
    B = MData[3]
    e = p / (gamma - 1) + 0.5 * rho * V**2 + 0.5 * B**2 / mu
    de_dt = -grad_x((e + p) * V) - grad_y((e + p) * V)
    return np.array(de_dt).ravel()

def F(t,x):
    return np.concatenate([continuity(t,x), momentum(t,x), energy(t,x), induction(t,x)])
    
def solve(Duration, resolution, rho_0, V_0, p_0, B_0):
    Time = np.linspace(0,Duration, int(Duration/resolution))
    Data = np.concatenate([np.array(rho_0).ravel(), np.array(V_0).ravel(), np.array(p_0).ravel(), np.array(B_0).ravel()])
    sol = solve_ivp(F,(0, Duration),Data)
    return sol
