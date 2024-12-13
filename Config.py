import numpy as np
# Физические константы
k_B = 1.38e-23  # Постоянная Больцмана (Дж/К)
m_p = 1.67e-27  # Масса протона (кг)
e = 1.6e-19  # Заряд электрона (Кл)
mu_0 = 4 * np.pi * 1e-7  # Магнитная проницаемость вакуума (Гн/м)
gamma = 5 / 3  # Адиабатический индекс
mu = 1.0                 # Магнитная проницаемость
eta = 0.01               # Магнитная диффузия
nu = 0                # Вязкость
m_D = 3.3437e-27 
m_T = 5.0075e-27
m_He = 6.6449e-27
Lx, Ly = 1.0, 1.0  # Размеры области по x и y
Nx, Ny = 50, 50   # Число точек сетки
dx, dy = Lx / Nx, Ly / Ny
def MevToK(mev):
    return 11604525006.17*mev
def concatenate_arrays(drho, dV_magnitude, dp, dB, dT, dn_D, dn_T, dn_He):
    """
    Concatenate multiple 2D arrays into a single 1D array.

    Parameters:
    - drho: Change in density (2D array)
    - dV_magnitude: Change in velocity magnitude (2D array)
    - dp: Change in pressure (2D array)
    - dB: Change in magnetic field (2D array)
    - dT: Change in temperature (2D array)
    - dn_D: Change in density of D (2D array)
    - dn_T: Change in density of T (2D array)
    - dn_He: Change in density of He (2D array)

    Returns:
    - A single 1D NumPy array containing all concatenated values.
    """
    return np.concatenate([
        drho.flatten(), 
        dV_magnitude.flatten(), 
        dp.flatten(), 
        dB.flatten(), 
        dT.flatten(), 
        dn_D.flatten(), 
        dn_T.flatten(), 
        dn_He.flatten()
    ])

def decode_arrays(concatenated_array, Nx, Ny):
    """
    Decode a 1D array back into multiple 2D arrays.

    Parameters:
    - concatenated_array: A single 1D NumPy array containing all values.
    - Nx: Number of rows.
    - Ny: Number of columns.

    Returns:
    - A tuple of eight 2D NumPy arrays corresponding to the original data.
    """
    # Calculate the size of each individual array
    size = Nx * Ny
    
    # Extract each array from the concatenated array
    drho = concatenated_array[0:size].reshape((Nx, Ny))
    dV_magnitude = concatenated_array[size:size*2].reshape((Nx, Ny))
    dp = concatenated_array[size*2:size*3].reshape((Nx, Ny))
    dB = concatenated_array[size*3:size*4].reshape((Nx, Ny))
    dT = concatenated_array[size*4:size*5].reshape((Nx, Ny))
    dn_D = concatenated_array[size*5:size*6].reshape((Nx, Ny))
    dn_T = concatenated_array[size*6:size*7].reshape((Nx, Ny))
    dn_He = concatenated_array[size*7:size*8].reshape((Nx, Ny))

    return drho, dV_magnitude, dp, dB, dT, dn_D, dn_T, dn_He

def decode_solve_ivp_result(sol, Nx, Ny):
    """
    Decode the result from solve_ivp into individual 2D arrays.

    Parameters:
    - sol: The solution object returned by scipy.integrate.solve_ivp.
    - Nx: Number of rows (grid points in x-direction).
    - Ny: Number of columns (grid points in y-direction).

    Returns:
    - A tuple of eight 2D NumPy arrays corresponding to the original data:
      (density_change, velocity_magnitude_change, pressure_change, 
       magnetic_field_change, temperature_change, n_D_change, n_T_change, n_He_change)
    """
    # The solution object contains an attribute 'y' which is the concatenated array
    concatenated_array = sol.y.flatten()  # Flatten to ensure it's a 1D array

    # Calculate the size of each individual array
    size = Nx * Ny
    
    # Extract each array from the concatenated array
    density_change = concatenated_array[0:size].reshape((Nx, Ny))
    velocity_magnitude_change = concatenated_array[size:size*2].reshape((Nx, Ny))
    pressure_change = concatenated_array[size*2:size*3].reshape((Nx, Ny))
    magnetic_field_change = concatenated_array[size*3:size*4].reshape((Nx, Ny))
    temperature_change = concatenated_array[size*4:size*5].reshape((Nx, Ny))
    n_D_change = concatenated_array[size*5:size*6].reshape((Nx, Ny))
    n_T_change = concatenated_array[size*6:size*7].reshape((Nx, Ny))
    n_He_change = concatenated_array[size*7:size*8].reshape((Nx, Ny))

    return (density_change, velocity_magnitude_change, pressure_change,
            magnetic_field_change, temperature_change, n_D_change,
            n_T_change, n_He_change)
def extract_frames_from_solution(sol, Nx, Ny):
    """
    Extracts frames of data from the solution object returned by solve_ivp.

    Parameters:
    - sol: The solution object returned by scipy.integrate.solve_ivp.
    - Nx: Number of rows (grid points in x-direction).
    - Ny: Number of columns (grid points in y-direction).

    Returns:
    - A dictionary containing arrays of shape (num_frames, Nx, Ny) for each variable.
    """
    # Number of time steps
    num_frames = sol.y.shape[1]

    # Prepare arrays to hold frames for each variable
    density_frames = np.zeros((num_frames, Nx, Ny))
    velocity_magnitude_frames = np.zeros((num_frames, Nx, Ny))
    pressure_frames = np.zeros((num_frames, Nx, Ny))
    magnetic_field_frames = np.zeros((num_frames, Nx, Ny))
    temperature_frames = np.zeros((num_frames, Nx, Ny))
    n_D_frames = np.zeros((num_frames, Nx, Ny))
    n_T_frames = np.zeros((num_frames, Nx, Ny))
    n_He_frames = np.zeros((num_frames, Nx, Ny))

    # Loop through each time step and extract data
    for i in range(num_frames):
        # Get the concatenated array for this time step
        concatenated_array = sol.y[:, i]
        
        # Decode the concatenated array into individual variables
        size = Nx * Ny
        density_frames[i] = concatenated_array[0:size].reshape((Nx, Ny))
        velocity_magnitude_frames[i] = concatenated_array[size:size*2].reshape((Nx, Ny))
        pressure_frames[i] = concatenated_array[size*2:size*3].reshape((Nx, Ny))
        magnetic_field_frames[i] = concatenated_array[size*3:size*4].reshape((Nx, Ny))
        temperature_frames[i] = concatenated_array[size*4:size*5].reshape((Nx, Ny))
        n_D_frames[i] = concatenated_array[size*5:size*6].reshape((Nx, Ny))
        n_T_frames[i] = concatenated_array[size*6:size*7].reshape((Nx, Ny))
        n_He_frames[i] = concatenated_array[size*7:size*8].reshape((Nx, Ny))

    return {
        'density': density_frames,
        'velocity_magnitude': velocity_magnitude_frames,
        'pressure': pressure_frames,
        'magnetic_field': magnetic_field_frames,
        'temperature': temperature_frames,
        'n_D': n_D_frames,
        'n_T': n_T_frames,
        'n_He': n_He_frames
    }

# Example usage after solving with solveMe
# sol = solveMe(duration, initial_data)
# frames = extract_frames_from_solution(sol, Nx, Ny)

# Access individual frames like this:
# density_frame_at_t0 = frames['density'][0]
# velocity_frame_at_t1 = frames['velocity_magnitude'][1]

# Example usage after solving with solveMe
# sol = solveMe(duration, initial_data)
# decoded_arrays = decode_solve_ivp_result(sol, Nx, Ny)