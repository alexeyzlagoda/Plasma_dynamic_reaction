from Config import *
import numpy as np
import scipy.integrate
from TempDestribution import *
from tqdm import tqdm

# Apply boundary conditions (closed walls)
def apply_boundary_conditions(density:np.array, velocity:np.array, magnetic_field:np.array, pressure:np.array):
    density[0, :] = density[-1, :] = 0
    velocity[0, :] = velocity[-1, :] = 0
    magnetic_field[0, :] = magnetic_field[-1, :] = 0
    pressure[0, :] = pressure[-1, :] = 0
    density[:, 0] = density[:, -1] = 0
    velocity[:, 0] = velocity[:, -1] = 0
    magnetic_field[:, 0] = magnetic_field[:, -1] = 0
    pressure[:, 0] = pressure[:, -1] = 0
    return density, velocity, magnetic_field, pressure

# Spatial derivatives
def grad_x(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def grad_y(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

def laplacian(f):
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 + \
           (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2

def continuity(density:np.array, velocity:np.array) ->np.array:
    return - (grad_x(density * velocity) + grad_y(density * velocity))

# Momentum equation
def momentum(rho_matrix:np.array, speed_matrix:np.array, p_matrix:np.array, H_matrix:np.array) -> np.array:
    du_dx = grad_x(speed_matrix)
    du_dy = grad_y(speed_matrix)
    
    dp_dx = grad_x(p_matrix)
    dp_dy = grad_y(p_matrix)

    laplace_u = laplacian(speed_matrix)

    # Lorentz force components
    dH_dx = grad_x(H_matrix)
    dH_dy = grad_y(H_matrix)
    
    lorentz_x = H_matrix * dH_dx - grad_x(0.5 * H_matrix**2)
    lorentz_y = H_matrix * dH_dy - grad_y(0.5 * H_matrix**2)

    # Time derivatives of velocity components
    du_dt = (-speed_matrix * (du_dx + du_dy) 
             - (1 / rho_matrix) * dp_dx 
             + mu * laplace_u 
             + lorentz_x / rho_matrix)
    
    dv_dt = (-speed_matrix * (du_dx + du_dy) 
             - (1 / rho_matrix) * dp_dy 
             + mu * laplacian(speed_matrix) 
             + lorentz_y / rho_matrix)

    return np.sqrt(du_dt**2 + dv_dt**2)

# Induction equation
def induction(magnetic_field:np.array, velocity:np.array)->np.array:
    return grad_x(velocity * magnetic_field) - grad_y(velocity * magnetic_field) + eta * laplacian(magnetic_field)

def F(popugay, IncData:np.array) -> np.array:    
    density, velocity, pressure, magnetic_field, temperature, n_D, n_T , n_He = decode_arrays(IncData, Nx, Ny)

    dT, dn_D, dn_T , dn_He  = UpdateTemp(temperature, n_D, n_T , n_He)
    
    dB = induction(magnetic_field, velocity)
    dp = -grad_x(density * velocity) - grad_y(density * velocity)
    
    dV_magnitude = momentum(density, velocity, pressure, magnetic_field)
    
    drho = continuity(density, velocity)

    # Concatenate results into a single array for integration step
    x_concatenated = concatenate_arrays(drho, dV_magnitude, dp, dB, dT, dn_D, dn_T, dn_He)
    return x_concatenated

def solveMe(duration, data:np.array):    
    sol = scipy.integrate.solve_ivp(F, duration, data)
    return sol
