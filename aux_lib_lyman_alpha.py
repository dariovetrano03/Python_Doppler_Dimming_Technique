import numpy as np
import ChiantiPy.core as ch

import matplotlib.pyplot as plt
from numba import njit



# --- PHYSICAL CONSTANTS ---

C_LIGHT_CGS = 2.9979e10   # [cm/s] Speed of light in vacuum
K_B_CGS = 1.3807e-16       # [erg/K] Boltzmann's constant
G_GRAVITATION_CGS = 6.674e-8 # [cm³ g⁻¹ s⁻²] Gravitational constant
H_PLANCK_CGS = 6.6261e-27 # [erg*s] Planck's constant

M_H_CGS = 1.6736e-24      # [g] Hydrogen atom mass
M_P_CGS = 1.6725e-24         # [g] Proton mass
R_SUN_CM = 6.957e10           # [cm] Solar radius
M_SUN_CGS = 1.989e33      # [g] Solar mass

LAMBDA0_NM = 121.567             # [nm] Wavelength of the Ly-alpha transition  
LAMBDA0_CM = LAMBDA0_NM * 1e-7        # [cm]
LAMBDA0_ANG = 1215.67 # [Å] 

NU0_HZ = C_LIGHT_CGS / LAMBDA0_CM # [Hz] Frequency of the Ly-alpha transition 

B12_CGS = 8.4838e9 # [ster * cm2 / phot / s] Einstein's absorption coefficient for the Ly-alpha transition (from Giordano et al., 2025)

J_CHROM_PHOT = 7e15 # [phot / cm2 / s / ster] Average intensity of the solar disk (from Giordano et al., 2025; Dolei et al., 2019)

SUM_Ai = 1.878 - 1.188 + 0.31017 # [-] ~ 1 (from Auchére, 2005)

""""
ASSUMING A CONSTANT T_HI:
"""
T_HI = 1e6 # [K]
"""
IMPROVEMENT: USE THE T_HI MAP FROM Dolei et al., 2018, Fig. 6
"""
"""
ASSUMING A CONSTANT T_HI IMPLIES THAT w_cm IS CONSTANT AS WELL:
"""

w_cm = (LAMBDA0_CM / C_LIGHT_CGS) * np.sqrt(2 * K_B_CGS * T_HI / M_H_CGS) # [cm]

"""
"""

A_He = 0.1 # [-] Helium abundance. A_He = 10% according to Withbroe et al. (1982), or A_He = 2.5% according to Moses et al. (2020)
n_pe = 1 / (2 * A_He + 1) # [-]  ratio of the proton number density (n_p) to the electron number density (n_e)



# Parameters for the tri-gaussian chromospheric profile: 'a' [-], 'delta_lambda' [cm], 'sigma' [cm] (from Auchère, 2005)
params = {'1': {'a': 1.878, 'delta_lambda': 0.001859e-7, 'sigma': 0.03075e-7},
          '2': {'a': -1.188, 'delta_lambda': 0.002087e-7, 'sigma': 0.02390e-7},
          '3': {'a': 0.31017, 'delta_lambda': 0.002159e-7, 'sigma': 0.07353e-7}}

p_a = np.array([p['a'] for p in params.values()])
p_delta_lambda = np.array([p['delta_lambda'] for p in params.values()])
p_sigma = np.array([p['sigma'] for p in params.values()])


@njit(fastmath=True, cache=True)
def trapz_numba(y, x):
    """A simple, Numba-compatible trapezoidal rule integrator."""
    s = 0.0
    for i in range(len(x) - 1):
        s += (x[i+1] - x[i]) * (y[i+1] + y[i])
    return s / 2.0

def get_hydrogen_neutral_fraction(temperatures_K):
    """
    Calculates the neutral hydrogen (H I) fraction in ionization equilibrium.

    This function uses the `ChiantiPy` library to calculate the fraction of
    hydrogen atoms that are in the neutral state (H I) at one or more
    specified temperatures. The calculation is based on the collisional
    ionization equilibrium model, using atomic data from the CHIANTI database.

    Args:
        temperatures_K (float | np.ndarray):
            The temperature or an array of temperatures in Kelvin [K] for which
            to perform the calculation.

    Returns:
        float | np.ndarray:
            The neutral hydrogen fraction corresponding to the input temperature(s).
            If the input was a single float, it returns a float. If the input
            was an array, it returns a NumPy array of fractions. The value is
            dimensionless and ranges between 0 and 1.

    Reference:
            - K. P. Dere, E. Landi, H.E. Mason, B. C. Monsignori Fossi, P. R. Young
              Astronomy and Astrophysics Suppl. Ser., Vol. 125, pp. 149-173 (1997)
            - R. P. Dufresne, G. Del Zanna, P. R. Young, K. P. Dere, E. Deliporanidou, W. T. Barnes, E. Landi
              The Astrophysical Journal, Volume 974, Issue 1, id.71, 15 pp. (October 2024)
             
    """
    temps_input = np.atleast_1d(temperatures_K)
    is_single_value = temps_input.size == 1

    temps_for_calc = np.array([temps_input[0], temps_input[0]]) if is_single_value else temps_input

    H_ioneq = ch.ioneq('h')
    H_ioneq.calculate(temperature=temps_for_calc)

    ion_balance = H_ioneq.Ioneq

    neutral_fraction_results = ion_balance[0]

    return neutral_fraction_results[0] if is_single_value else neutral_fraction_results

def get_Gibson_temperature(r_solar):
    """
    Calculates electron temperature from the hydrostatic model of Gibson et al. (1999).

    This function implements the spherically symmetric, hydrostatic model for the
    solar corona as described in Gibson et al. (1999). It computes the electron
    temperature (T_e) at a given radial distance by first calculating the
    electron density and pressure profiles based on empirical fits to observations.

    Args:
        r_solar (float or np.ndarray):
            The radial distance from the Sun's center, in units of solar radii.

    Returns:
        float or np.ndarray:
            The calculated electron temperature in Kelvin [K].

    Notes:
        This function relies on two globally defined constants:
        - `R_SUN_CM`: The solar radius in centimeters.
        - `K_B_CGS`: The Boltzmann constant in CGS units (erg/K).

    Reference:
        Gibson, S. E., Fludra, A., Bagenal, F., et al. 1999, J. Geophys. Res., 104, 9691
    """
    
    # --- Model parameters from Gibson et al. (1999), Table 1 ---
    alpha = 0.1           # [-] Helium abundance (n_He / n_p)
    a, b = 3.60, 15.3     # Parameters for the power-law fit of density
    c, d = 0.990, 7.34
    e, f = 0.365, 4.31

    # --- Calculations ---
    # Electron density n_e(r) from the three-part power law (Eq. 4 in the paper)
    # The 1e8 factor converts the model's base units to cm^-3.
    ne_r = (a * r_solar**-b + c * r_solar**-d + e * r_solar**-f) * 1e8

    # Pre-factor for the pressure integral, derived from the hydrostatic equilibrium equation.
    K_pressure = ((1 + 4*alpha) / (1 + 2*alpha)) * G_GRAVITATION_CGS * M_SUN_CGS * M_P_CGS * 1e8

    # Pressure P(r) obtained by integrating the density profile from r to infinity (Eq. 3).
    P_r = K_pressure * ((a/(b+1))*r_solar**-(b+1) + (c/(d+1))*r_solar**-(d+1) + (e/(f+1))*r_solar**-(f+1))
    P_r /= R_SUN_CM 

    temperature = ((1 + 2*alpha) / (2 + 3*alpha)) * (P_r / (ne_r * K_B_CGS))

    return temperature


""""""
    
def precompute_los_arrays(r_pos_rsun, ne_interpolator, r_max, num_points_los=100):
    x_los_rsun = np.linspace(-2 * r_max, 2 * r_max, num_points_los)
   
    r_3d_rsun = np.sqrt(r_pos_rsun**2 + x_los_rsun**2)
   
    n_e_array = ne_interpolator(r_3d_rsun)
    Te_array = get_Gibson_temperature(r_3d_rsun) 
    Ri_array= get_hydrogen_neutral_fraction(Te_array)
  
    beta_array = np.arccos(np.clip(r_pos_rsun / r_3d_rsun, -1, 1))
    x_los_cm = x_los_rsun * R_SUN_CM
    
    r_cm = r_3d_rsun * R_SUN_CM # [R☉] → [cm]
    
    theta_max = np.arcsin(R_SUN_CM / r_cm) # [rad]

    theta_matrix = np.linspace(0, theta_max, 100) # [rad]

    theta_matrix = theta_matrix.T

    
    return r_3d_rsun, n_e_array, Ri_array, beta_array, x_los_cm, theta_matrix

def precompute_los_arrays_II(r_pos_rsun, ne_interpolator, Vw_interpolator, r_max, num_points_los=100):

    x_los_rsun = np.linspace(-2 * r_max, 2 * r_max, num_points_los)
   
    r_3d_rsun = np.sqrt(r_pos_rsun**2 + x_los_rsun**2)
   
    n_e_array = ne_interpolator(r_3d_rsun)
    
    Vw_array = Vw_interpolator(r_3d_rsun)

    Vw_array[Vw_array > 500] = 500
    
    Te_array = get_Gibson_temperature(r_3d_rsun) 

    Ri_array= get_hydrogen_neutral_fraction(Te_array)
  
    beta_array = np.arccos(np.clip(r_pos_rsun / r_3d_rsun, -1, 1))
    
    x_los_cm = x_los_rsun * R_SUN_CM
    
    r_cm = r_3d_rsun * R_SUN_CM # [R☉] → [cm]
    
    theta_max = np.arcsin(R_SUN_CM / r_cm) # [rad]

    theta_matrix = np.linspace(0, theta_max, 100) # [rad]

    theta_matrix = theta_matrix.T
    
    return r_3d_rsun, n_e_array, Vw_array, Ri_array, beta_array, theta_matrix, x_los_cm



@njit(fastmath=True, cache=True)
def calculate_emissivity_numba(vw_cms, n_e, R_i, beta_rad, theta_arr, 
                                 p_a, p_delta_lambda, p_sigma):
    """
    Numba-jitted "kernel" for a SINGLE point on the LOS.
    This function has not changed.
    """
    # Angular term calculation
    cos2_beta = np.cos(beta_rad)**2
    sin2_beta = 1.0 - cos2_beta
    cos2_theta = np.cos(theta_arr)**2
    sin2_theta = 1.0 - cos2_theta
    angular_term = (11.0 + 3.0 * (cos2_beta * cos2_theta + 0.5 * sin2_beta * sin2_theta)) / 12.0

    # Sum term calculation (explicit loop)
    sum_term = np.zeros_like(theta_arr)
    for i in range(p_a.shape[0]):
        delta_lambda_i = p_delta_lambda[i]
        sigma_i = p_sigma[i]
        doppler_shift = (LAMBDA0_CM / C_LIGHT_CGS) * vw_cms * np.cos(theta_arr)
        numerator = -(delta_lambda_i - doppler_shift)**2
        denominator = w_cm**2 + sigma_i**2
        sum_term += (p_a[i] / np.sqrt(denominator)) * np.exp(numerator / denominator)

    integrand = angular_term * np.sin(theta_arr) * sum_term
    integral_theta = trapz_numba(integrand, theta_arr)

    prefactor = (n_pe * B12_CGS * H_PLANCK_CGS * LAMBDA0_CM * J_CHROM_PHOT) / (4 * np.pi * SUM_Ai * 2 * np.sqrt(np.pi))
    
    return prefactor * n_e * R_i * integral_theta

def integrate_intensity_los_numba(vw_kms, n_e_array, Ri_array, beta_array,
                                      theta_matrix, x_los_cm,
                                      p_a, p_delta_lambda, p_sigma):
    """
    This is a regular Python function that acts as a fast dispatcher
    to the appropriate Numba-compiled kernel.
    """
    vw_cms = vw_kms * 1e5

    # This 'if' happens once, in normal Python. The overhead is negligible.
    if isinstance(vw_cms, (float, int)):
        # Call the specialized Numba function for constant velocity
        return _integrate_los_const_v_numba(
            vw_cms, n_e_array, Ri_array, beta_array, theta_matrix,
            x_los_cm, p_a, p_delta_lambda, p_sigma
        )
    else:
        # Call the specialized Numba function for a velocity profile
        return _integrate_los_profile_v_numba(
            vw_cms, n_e_array, Ri_array, beta_array, theta_matrix,
            x_los_cm, p_a, p_delta_lambda, p_sigma
        )


@njit(fastmath=True, cache=True)
def _integrate_los_const_v_numba(vw_cms, n_e_array, Ri_array, beta_array,
                                  theta_matrix, x_los_cm,
                                  p_a, p_delta_lambda, p_sigma):
    """Numba kernel ONLY for a constant velocity (float)."""
    emissivity_values = np.zeros(n_e_array.shape[0], dtype=np.float64)
    for i in range(n_e_array.shape[0]):
        # No 'if' check needed, vw_cms is always the same float.
        emissivity_values[i] = calculate_emissivity_numba(
            vw_cms, n_e_array[i], Ri_array[i], beta_array[i],
            theta_matrix[i], p_a, p_delta_lambda, p_sigma
        )
    return trapz_numba(emissivity_values, x_los_cm)

@njit(fastmath=True, cache=True)
def _integrate_los_profile_v_numba(vw_cms, n_e_array, Ri_array, beta_array,
                                    theta_matrix, x_los_cm,
                                    p_a, p_delta_lambda, p_sigma):
    """Numba kernel ONLY for a velocity profile (array)."""
    emissivity_values = np.zeros(n_e_array.shape[0], dtype=np.float64)
    for i in range(n_e_array.shape[0]):
        # No 'if' check needed, vw_cms is always an array to be indexed.
        emissivity_values[i] = calculate_emissivity_numba(
            vw_cms[i], n_e_array[i], Ri_array[i], beta_array[i],
            theta_matrix[i], p_a, p_delta_lambda, p_sigma
        )
    return trapz_numba(emissivity_values, x_los_cm)