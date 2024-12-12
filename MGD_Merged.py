from Config import *
import numpy as np
from scipy.integrate import solve_ivp
from TempDestribution import *
from tqdm import tqdm

# Граничные условия (замкнутые стенки)
def apply_boundary_conditions(dencity:np.array, velocity:np.array, magnetic_field:np.array, pressure:np.array):
    for var in [dencity, velocity, magnetic_field, pressure]:
        var[0, :] = var[-1, :] = 0
        var[:, 0] = var[:, -1] = 0
    return dencity, velocity, magnetic_field, pressure

# Производные по пространству
def grad_x(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def grad_y(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

def laplacian(f):
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 + \
           (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2

def continuity(dencity:np.array, velocity:np.array) -> np.array:
    drho_dt = - (grad_x(dencity * velocity) + grad_y(dencity * velocity))
    return drho_dt

# Уравнение движения
def momentum(dencity:np.array, velocity:np.array, pressure:np.array, magnetic_field:np.array)-> np.array:
    dV_dt = - velocity * grad_x(velocity) - velocity * grad_y(velocity) \
            - grad_x(pressure) / dencity + nu * laplacian(velocity)
    return dV_dt

# Уравнение индукции
def induction(magnetic_field:np.array, velocity:np.array)-> np.array:
    dB_dt = grad_x(velocity * magnetic_field) - grad_y(velocity * magnetic_field) + eta * laplacian(magnetic_field)
    return dB_dt

# Уравнение энергии
def energy(dencity:np.array, velocity:np.array, pressure:np.array, magnetic_field:np.array)-> np.array:
    e = pressure / (gamma - 1) + 0.5 * dencity * velocity**2 + 0.5 * magnetic_field**2 / mu
    de_dt = -grad_x((e + pressure) * velocity) - grad_y((e + pressure) * velocity)
    return de_dt

def F(Temperature,IncData)-> np.array:
    dencity, velocity, pressure =  np.array(),np.array(),np.array()
    magnetic_field, Temperature, n_D  = np.array(),np.array(),np.array()
    n_T , n_He = np.array(), np.array()
    for var, n  in zip([dencity, velocity, pressure, magnetic_field, Temperature, n_D,  n_T , n_He], range(8)):
        var = IncData.reshape((8,Nx, Ny))[n]
    dT, dn_D,  dn_T , dn_He  = UpdateTemp(Temperature, n_D,  n_T , n_He)
    dE = energy(dencity, velocity, pressure, magnetic_field) + 3 * k_B * Temperature
    dB = induction(magnetic_field, velocity)
    dV = momentum(dencity, velocity, pressure, magnetic_field)
    drho = continuity(dencity, velocity)
    return np.concatenate([drho.ravel(), dV.ravel(), np.zeros((Nx, Ny)).ravel(), dB.ravel(), dT.ravel(), dn_D.ravel(),  dn_T.ravel() , dn_He.ravel()])

def solveMe(Duration, Data):    
    sol = solve_ivp(F,(0, Duration),Data)
    return sol