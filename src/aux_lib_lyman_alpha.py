import numpy as np
import ChiantiPy.core as ch

import matplotlib.pyplot as plt
from numba import njit

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from src.metis_aux_lib import fit_negative_power_series, sqrt_model
from tqdm import tqdm




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

J_CHROM_PHOT = 5.5e15 # [phot / cm2 / s / ster] Average intensity of the solar disk (from Giordano et al., 2025; Dolei et al., 2019)

E_PHOTON_LYA_ERG = H_PLANCK_CGS * C_LIGHT_CGS / LAMBDA0_CM

J_CHROM_ERG = J_CHROM_PHOT * E_PHOTON_LYA_ERG

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

def precompute_los_arrays(r_pos_rsun, ne_interpolator, H_interpolator, r_max, num_points_los=100):    
# def precompute_los_arrays(r_pos_rsun, ne_interpolator, r_max, num_points_los=100):
    x_los_rsun = np.linspace(-2 * r_max, 2 * r_max, num_points_los)
   
    r_3d_rsun = np.sqrt(r_pos_rsun**2 + x_los_rsun**2)
   
    n_e_array = ne_interpolator(r_3d_rsun)
    Te_array = get_Gibson_temperature(r_3d_rsun) 
    Ri_array= H_interpolator(Te_array)
  
    beta_array = np.arccos(np.clip(r_pos_rsun / r_3d_rsun, -1, 1))
    x_los_cm = x_los_rsun * R_SUN_CM
    
    r_cm = r_3d_rsun * R_SUN_CM # [R☉] → [cm]
    
    theta_max = np.arcsin(R_SUN_CM / r_cm) # [rad]

    theta_matrix = np.linspace(0, theta_max, 100) # [rad]

    theta_matrix = theta_matrix.T

    
    return r_3d_rsun, n_e_array, Ri_array, beta_array, x_los_cm, theta_matrix

def precompute_los_arrays_II(r_pos_rsun, ne_interpolator, Vw_interpolator, H_interpolator, r_max, num_points_los=100):    

    x_los_rsun = np.linspace(-2 * r_max, 2 * r_max, num_points_los)
   
    r_3d_rsun = np.sqrt(r_pos_rsun**2 + x_los_rsun**2)
   
    n_e_array = ne_interpolator(r_3d_rsun)
    
    Vw_array = Vw_interpolator(r_3d_rsun)

    # Vw_array[Vw_array > 500] = 500
    
    Te_array = get_Gibson_temperature(r_3d_rsun) 

    Ri_array= H_interpolator(Te_array)
  
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
   
    # Correzione: LAMBDA0_CM va al denominatore e il 4*np.pi viene rimosso.
    # prefactor = (H_PLANCK_CGS * B12_CGS * n_pe * J_CHROM_ERG) / (SUM_Ai * 2 * np.sqrt(np.pi) * LAMBDA0_CM)
    
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








""""""""""""""""""""""""""""""""""""

def compute_velocity_maps(
    polar_ne_resam, polar_uv_resam, r_matrix, angle_ROI=None, 
    num_points_los=100, save_matrix=False, date=None, save_path=None
):
    """
    Compute the velocity maps (step I and II) from resampled electron density and UV intensity maps.
    
    Parameters
    ----------
    polar_ne_resam, polar_uv_resam : np.ndarray
        Resampled polar maps.
    r_matrix : np.ndarray
        Radial coordinates.
    params : dict
        Dictionary containing 'a', 'delta_lambda', 'sigma' for each line.
    angle_ROI : array-like, optional
        Angles to process [deg]. Default: 0-359.
    num_points_los : int
        Points along LOS integration.
    save_matrix : bool
        If True, saves velocity maps as .npy.
    date_pb : str
        Date string for filename (required if save_matrix=True).
        
    Returns
    -------
    velocity_map_kms_I, velocity_map_kms_II : np.ndarray
        Computed velocity maps.
    """
    
    if angle_ROI is None:
        angle_ROI = np.arange(360)
    
    # Extract CHIANTI parameters
    p_a = np.array([p['a'] for p in params.values()])
    p_delta_lambda = np.array([p['delta_lambda'] for p in params.values()])
    p_sigma = np.array([p['sigma'] for p in params.values()])

    # --- Step I: initial velocity map ---
    velocity_map_kms_I = np.zeros_like(polar_ne_resam)
    
    r_matrix_ROI = r_matrix[:, angle_ROI]
    total_pixels = np.sum(~np.isnan(r_matrix_ROI))
    current_pixel = 0

    # CHIANTI pre-computation
    r_min_glob = np.nanmin(r_matrix_ROI)
    r_max_glob = np.nanmax(r_matrix_ROI)
    r_3d_precomp = np.linspace(r_min_glob, r_max_glob * np.sqrt(5), 100)
    Te_array_precomp = get_Gibson_temperature(r_3d_precomp)

    H_ioneq = ch.ioneq('h')
    H_ioneq.calculate(temperature=Te_array_precomp)
    neutral_fraction_grid = H_ioneq.Ioneq[0]
    H_interpolator = interp1d(
        Te_array_precomp, neutral_fraction_grid,
        kind="linear", fill_value="extrapolate"
    )

    for i, angle in enumerate(tqdm(angle_ROI, desc="Step I: angles")):
        r_arr = r_matrix[:, angle][~np.isnan(polar_ne_resam[:, angle])]
        radial_ne = polar_ne_resam[~np.isnan(polar_ne_resam[:, angle]), angle]
        model_ne, _ = fit_negative_power_series(r_arr, radial_ne, degree=3)
        radial_interp_ne = model_ne
        r_max = r_arr[-1]

        for j, r_pos in enumerate(r_arr):
            current_pixel += 1
            I_obs = polar_uv_resam[j, angle]

            if np.isnan(I_obs):
                velocity_map_kms_I[j, angle] = np.nan
                continue

            _, n_e_arr, Ri_arr, beta_arr, x_cm, theta_mat = precompute_los_arrays(
                r_pos, radial_interp_ne, H_interpolator, r_max, num_points_los=num_points_los
            )

            I_static = integrate_intensity_los_numba(
                0, n_e_arr, Ri_arr, beta_arr, theta_mat, x_cm, p_a, p_delta_lambda, p_sigma
            )

            def objective_function(vw_kms):
                return integrate_intensity_los_numba(
                    vw_kms, n_e_arr, Ri_arr, beta_arr, theta_mat, x_cm, p_a, p_delta_lambda, p_sigma
                ) - I_obs

            if I_static > I_obs and not np.isclose(I_static, I_obs, rtol=5e-2):
                try:
                    found_velocity = brentq(objective_function, a=0, b=500, rtol=0.01)
                    velocity_map_kms_I[j, angle] = found_velocity
                except ValueError:
                    velocity_map_kms_I[j, angle] = np.nan
            else:
                if np.isclose(I_static, I_obs, rtol=5e-2):
                    velocity_map_kms_I[j, angle] = 0
                else:
                    velocity_map_kms_I[j, angle] = np.nan

    # --- Step II: refined velocity map ---
    velocity_map_kms_II = np.zeros_like(polar_ne_resam)
    current_pixel = 0
    total_pixels = np.sum(~np.isnan(r_matrix[:, angle_ROI]))

    for i, angle in enumerate(tqdm(angle_ROI, desc="Step II: angles")):
        # Radial electron density model
        r_arr_ne = r_matrix[~np.isnan(polar_ne_resam[:, angle]), angle]
        radial_ne = polar_ne_resam[~np.isnan(polar_ne_resam[:, angle]), angle]
        model_ne, _ = fit_negative_power_series(r_arr_ne, radial_ne, degree=3)
        radial_interp_ne = model_ne

        # Velocity model from Step I
        mask_Vw = ~np.isnan(velocity_map_kms_I[:, angle]) & ~np.isnan(r_matrix[:, angle])
        r_arr_Vw = r_matrix[mask_Vw, angle]
        radial_Vw = velocity_map_kms_I[mask_Vw, angle]

        popt, _ = curve_fit(sqrt_model, r_arr_Vw, radial_Vw, p0=[10, 0], bounds=([-np.inf, 0], [np.inf, np.inf]))
        a, b = popt
        radial_interp_Vw = lambda x: a + b * np.sqrt(x)

        r_max = r_arr_Vw[-1]

        for j, r_pos in enumerate(r_arr_Vw):
            current_pixel += 1
            I_obs = polar_uv_resam[j, angle]

            if np.isnan(I_obs):
                velocity_map_kms_II[j, angle] = np.nan
                continue

            r_los, n_e_los, Vw_los, Ri_los, beta_los, theta_mat, x_cm = precompute_los_arrays_II(
                r_pos, radial_interp_ne, radial_interp_Vw, H_interpolator, r_max
            )


            """ OPTION 1 for dimming: PROFILE SCALING"""
            # def objective_function(Vw_pos_kms):
            #     scaled_Vw_profile = Vw_pos_kms * Vw_los / radial_interp_Vw(r_pos)
            #     return integrate_intensity_los_numba(
            #         scaled_Vw_profile, n_e_arr, Ri_arr, beta_arr, theta_mat, x_cm,
            #         p_a, p_delta_lambda, p_sigma
            #     ) - I_obs

            """ OPTION 2 for dimming: PROFILE SHIFTING"""
            def objective_function(Vw_pos_kms):
                delta_V = Vw_pos_kms - radial_interp_Vw(r_pos)
                shifted_Vw_profile = Vw_los + delta_V
                return integrate_intensity_los_numba(shifted_Vw_profile, n_e_los, Ri_los, beta_los, theta_mat, x_cm, p_a, p_delta_lambda, p_sigma) - I_obs

            try:
                found_velocity = brentq(objective_function, a=0, b=500, rtol=0.01)
                velocity_map_kms_II[j, angle] = found_velocity
            except ValueError:
                velocity_map_kms_II[j, angle] = 0

    # --- Save matrices if requested ---
    if save_matrix:
        if date is None:
            raise ValueError("date_pb must be provided when save_matrix=True")
        np.save(f"{save_path}/vw_map_step_1_{date.replace(':', '')}.npy", velocity_map_kms_I)
        print("Velocity maps saved.")

    np.save(f"{save_path}/vw_map_step_2_sqrt_{date.replace(':', '')}.npy", velocity_map_kms_II)

    return velocity_map_kms_I, velocity_map_kms_II
