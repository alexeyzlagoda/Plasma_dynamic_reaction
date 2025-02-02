import sys 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Config as CFG
import MGD_Merged as MGD

def RunSim(Time, n_D, n_T, n_He, rho, V, p, B, Temp):
    x = CFG.concatenate_arrays(rho, V, p, B, Temp, n_D, n_T, n_He)
    sol = MGD.solveMe(Time, x)
    Recieved_Data = CFG.extract_frames_from_solution(sol,CFG.Nx, CFG.Ny)
    return Recieved_Data

def CheckAvaliableReaction(Recieved_Data): 
    Average_Temp_Time:np.array
    for frame in Recieved_Data:
        Average_Temp_Time.append(np.mean(frame))
    return Average_Temp_Time

    