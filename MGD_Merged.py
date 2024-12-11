from Config import *
from MGD import *
from Solver1 import *
import numpy as np
from scipy.integrate import solve_ivp

def F(t,IncData):
    MData = np.array(IncData).reshape(8, Nx, Ny)

    rho = MData[0];V = MData[1]; p = MData[2];B = MData[3]; Temp = MData[4]; n_D = MData[5]; n_T = MData[6]; n_He = MData[7]

    x =  np.concatenate([np.array(rho).ravel(), np.array(V).ravel(), np.array(p).ravel(), np.array(B).ravel()])
    x =  np.concatenate([continuity(t,x), momentum(t,x), energy(t,x), induction(t,x)])

    MData = np.array(IncData).reshape(4, Nx, Ny)
    rho = MData[0]
    V = MData[1]
    p = MData[2]
    B = MData[3]
    Temp, n_D, n_T, n_He = UpdateTemp(Temp,n_D, n_T, n_He)
    V = np.sqrt(3 * k_B * Temp / (np.sum(n_D) * m_D + np.sum(n_T) * m_T + np.sum(n_He) * m_He) )
    return np.concatenate([np.array(rho).ravel(), np.array(V).ravel(), np.array(p).ravel(), np.array(B).ravel(), np.array(T).ravel(), np.array(n_D).ravel(), np.array(n_T).ravel(), np.array(n_He).ravel()])

def solve(Duration, rho_0, V_0, p_0, B_0, T_0, n_D_0, n_T_0, n_He_0):
    Data = np.concatenate([np.array(rho_0).ravel(), np.array(V_0).ravel(), np.array(p_0).ravel(), np.array(B_0).ravel(), np.array(T_0).ravel(), np.array(n_D_0).ravel(), np.array(n_T_0).ravel(), np.array(n_He_0).ravel()])
    sol = solve_ivp(F,(0, Duration),Data)
    return sol