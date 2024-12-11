import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Config import *
from Struct import *

def fusion_rate(n_D, n_T, T_e):
    """Скорость реакций слияния (дейтерий-дейтерий и дейтерий-тритий)."""
    reaction_rate_DD = 2.8e-14 * (n_D**2) * np.exp(-1.0 / T_e)
    reaction_rate_DT = 2.8e-14 * (n_D*n_T) * np.exp(-1.0 / T_e)
    # print(n_D)
    # print(n_T)
    R_DD = reaction_rate_DD * n_D * n_D
    R_DT = reaction_rate_DT * n_D * n_T
    return R_DD, R_DT

def UpdateTemp(T_e, n_D, n_T,n_He):
    R_DD, R_DT = fusion_rate(n_D, n_T, T_e)
    # for i in R_DD:
    #     print(*i)
    # print()
    # for i in R_DT:
    #     print(*i)
    # print()
    n_D, n_T, n_He = n_D - (2* R_DD + R_DT), n_T - R_DT, R_DT + n_He
    return R_DT*MevToK(17.6) + R_DD * MevToK(7.3), n_D, n_T, n_He

def SolvMain(V, dt, T_e, n_D, n_T, n_He, B, p):
    pass    
    
def Sim(Config:SolverConf):
    pass