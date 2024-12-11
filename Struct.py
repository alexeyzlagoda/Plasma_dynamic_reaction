import numpy as np
from Config import *
class SolverConf:
    Lx, Ly = 1.0,1.0
    Nx, Ny = 1000, 1000  # Число ячеек сетки
    n_D = np.ones((Nx, Ny)) * 1e19  # Плотность дейтерия (м^-3)
    n_T = np.ones((Nx, Ny)) * 1e19  # Плотность трития (м^-3)
    n_He = np.zeros((Nx, Ny))       # Плотность гелия (м^-3)
    T_e = np.ones((Nx, Ny)) * 1e8   # Температура (К)
    vx = np.zeros((Nx, Ny))         # Скорость по x (м/с)
    vy = np.zeros((Nx, Ny))         # Скорость по y (м/с)
    Bx = np.ones((Nx, Ny)) * 1.0    # Магнитное поле по x (Тл)
    By = np.zeros((Nx, Ny))         # Магнитное поле по y (Тл)
    P = n_D * k_B * T_e             # Давление (Па)

    source_D = np.zeros((Nx, Ny))
    source_T = np.zeros((Nx, Ny))
    sink = np.zeros((Nx, Ny))
    source_D[Nx // 4:Nx // 2, Ny // 4:Ny // 2] = 1e22
    source_T[Nx // 2:Nx // 4 * 3, Ny // 2:Ny // 4 * 3] = 1e22
    