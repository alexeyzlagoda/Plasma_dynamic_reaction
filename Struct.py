import numpy as np
from Config import *
class SolverObj:
    Nx, Ny = 50,50
    B    = np.zeros((Nx,Ny))
    rho  = np.zeros((Nx,Ny))
    Temp = np.zeros((Nx,Ny))
    p    = np.zeros((Nx,Ny))
    V    = np.zeros((Nx,Ny))
    n_D  = np.zeros((Nx,Ny))
    n_T  = np.zeros((Nx,Ny))
    n_He = np.zeros((Nx,Ny))
    E    = np.zeros((Nx,Ny))
    def __init__(self):
        pass 
    def __new__(self):
        pass
    def __sum__(self, A):
        self.B    += A.B   
        self.rho  += A.rho 
        self.Temp += A.Temp
        self.p    += A.p   
        self.V    += A.V   
        self.n_D  += A.n_D 
        self.n_T  += A.n_T 
        self.n_He += A.n_He
        self.E    += A.E    
    def __mul__(self,A):
        self.B    *= A
        self.rho  *= A
        self.Temp *= A
        self.p    *= A
        self.V    *= A
        self.n_D  *= A
        self.n_T  *= A
        self.n_He *= A
        self.E    *= A