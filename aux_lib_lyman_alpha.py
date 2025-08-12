import numpy as np
import ChiantiPy.core as ch
from scipy.integrate import trapz

import matplotlib.pyplot as plt

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

J_CHROM_PHOT = 5.78e15 # [phot / cm2 / s / ster] Average intensity of the solar disk (from Giordano et al., 2025; Dolei et al., 2019)

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
    c_fit, d = 0.990, 7.34
    e, f = 0.365, 4.31

    # --- Calculations ---
    # Electron density n_e(r) from the three-part power law (Eq. 4 in the paper)
    # The 1e8 factor converts the model's base units to cm^-3.
    ne_r = (a * r_solar**-b + c_fit * r_solar**-d + e * r_solar**-f) * 1e8

    # Pre-factor for the pressure integral, derived from the hydrostatic equilibrium equation.
    K_pressure = ((1 + 4*alpha) / (1 + 2*alpha)) * G_GRAVITATION_CGS * M_SUN_CGS * M_P_CGS * 1e8

    # Pressure P(r) obtained by integrating the density profile from r to infinity (Eq. 3).
    P_r = K_pressure * ((a/(b+1))*r_solar**-(b+1) + (c_fit/(d+1))*r_solar**-(d+1) + (e/(f+1))*r_solar**-(f+1))
    P_r /= R_SUN_CM 

    temperature = ((1 + 2*alpha) / (2 + 3*alpha)) * (P_r / (ne_r * K_B_CGS))

    return temperature


""""""
    
def precompute_los_arrays(r_pos_rsun, ne_interpolator, r_max, num_points_los=100):

    """
    Precomputes arrays needed for line-of-sight (LOS) integration in the solar corona.

    This function generates arrays representing the geometry and plasma parameters
    along the line of sight for a given heliocentric radial position (impact parameter). 
    It calculates the 3D radial distances, electron densities interpolated from a provided function,
    the angle between the LOS and the impact direction (beta), LOS distances in centimeters,
    and a range of scattering angles for resonant scattering computations (theta).

    Parameters
    ----------
    r_pos_rsun : float
        Radial position of the LOS in units of solar radii (R☉).
    ne_interpolator : callable
        Interpolating function that returns electron density given a radial distance in R☉.
    r_max : float
        Maximum radial distance along the LOS in units of solar radii.
    num_points_los : int, optional
        Number of points along the line of sight for discretization (default is 100).

    Returns
    -------
    r_3d_rsun : ndarray
        Array of 3D radial distances along the LOS in units of solar radii.
    n_e_array : ndarray
        Electron density values interpolated at the 3D radial distances.
    Ri_array : ndarray
        Neutral hydrogen fractions (dimensionless, between 0 and 1) along the LOS, representing the fraction of hydrogen atoms available for resonant scattering.
    beta_array : ndarray
        Array of angles (in radians) between the LOS direction and the impact vector.
    x_los_cm : ndarray
        Distances along the LOS converted from solar radii to centimeters.
    theta_matrix : ndarray
        Array of scattering angles (in radians) for integration over incident radiation directions.

    Notes
    -----
    - The LOS is assumed to be symmetric around the position r_pos_rsun, spanning from -2*r_max to 2*r_max.
    - The theta angles represent the scattering angle range for resonant scattering computations and
      are linearly spaced between 0 and the maximum angle determined by solar radius geometry.
    """
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

    Vw_array[Vw_array > 600] = 600
    
    Te_array = get_Gibson_temperature(r_3d_rsun) 

    Ri_array= get_hydrogen_neutral_fraction(Te_array)
  
    beta_array = np.arccos(np.clip(r_pos_rsun / r_3d_rsun, -1, 1))
    
    x_los_cm = x_los_rsun * R_SUN_CM
    
    r_cm = r_3d_rsun * R_SUN_CM # [R☉] → [cm]
    
    theta_max = np.arcsin(R_SUN_CM / r_cm) # [rad]

    theta_matrix = np.linspace(0, theta_max, 100) # [rad]

    theta_matrix = theta_matrix.T
    
    return r_3d_rsun, n_e_array, Vw_array, Ri_array, beta_array, theta_matrix, x_los_cm



def calculate_emissivity(r_3d, vw_cms, n_e, R_i, beta_rad, theta_rad):
    """
    Compute the local H I Lyman-alpha emissivity due to resonant scattering in the solar corona.

    This function calculates the specific emissivity (photons cm⁻³ s⁻¹ sr⁻¹) of
    neutral hydrogen atoms scattering chromospheric Lyman-alpha photons, based on
    physical models described in the literature (e.g., Dolei et al., 2019; Giordano et al., 2025).

    Parameters
    ----------
    r_3d : float
        Heliocentric distance of the scattering location (i.e., the point along the LOS) in solar radii (R☉).
    vw_cms : float
        Bulk solar wind velocity at the scattering location, in cm/s.
    n_e : float
        Local electron number density at the scattering location, in cm⁻³.
    R_i : float
        Neutral hydrogen fraction (dimensionless, between 0 and 1) at the scattering point, representing the fraction of hydrogen atoms available for resonant scattering.
    beta_rad : float
        Angle (in radians) between the line of sight and the impact direction at the scattering point.
    theta_rad : ndarray
        Array of scattering angles (in radians) over which the emissivity is integrated.


    Returns
    -------
    float
        Local specific emissivity of the Lyman-alpha line in units of photons cm⁻³ s⁻¹ sr⁻¹.

    Notes
    -----
    - The calculation involves integration over scattering angles weighted by the angular
      redistribution function and velocity-dependent Doppler shifts.
    - Constants and parameters such as transition probabilities and chromospheric intensity 
      are embedded in the implementation.
    """
    
    cos2_beta = np.cos(beta_rad)**2
    sin2_beta = np.sin(beta_rad)**2
    cos2_theta = np.cos(theta_rad)**2
    sin2_theta = np.sin(theta_rad)**2

    angular_term = (11.0 + 3.0 * (cos2_beta * cos2_theta + 0.5 * sin2_beta * sin2_theta)) / 12.0

    sum_term = np.zeros_like(theta_rad)
    
    for p in params.values():
        delta_lambda_i_cm = p['delta_lambda']
        sigma_i_cm = p['sigma']
        numerator = -(delta_lambda_i_cm - (LAMBDA0_CM / C_LIGHT_CGS) * vw_cms * np.cos(theta_rad))**2
        denominator = w_cm**2 + sigma_i_cm**2
        sum_term += (p['a'] / np.sqrt(denominator)) * np.exp(numerator / denominator)
        
    integrand = angular_term * np.sin(theta_rad) * sum_term

    integral_theta = trapz(integrand, theta_rad)

    # Eq. (A.10) from Giordano et al., 2025:
    prefactor = (n_pe * B12_CGS *  H_PLANCK_CGS * LAMBDA0_CM * J_CHROM_PHOT ) / (4 * np.pi * SUM_Ai * 2 * np.sqrt(np.pi)) 
    
    return prefactor * n_e * R_i * integral_theta


def integrate_intensity_los(vw_kms, r_3d_rsun, n_e_array, Ri_array, beta_array, theta_matrix, x_los_cm, T_HI=1e6):
    """
    Integrate the H I Lyman-alpha emissivity along the line of sight to compute observed intensity.

    This function calculates the total Lyman-alpha intensity observed along a line of sight
    by integrating the local emissivity, which is computed at discrete points along the LOS.

    Parameters
    ----------
    vw_kms : float
        Bulk solar wind velocity in km/s.
    r_3d_rsun : ndarray
        Array of heliocentric distances along the LOS in units of solar radii (R☉).
    n_e_array : ndarray
        Electron density values at each point along the LOS, in cm⁻³.
    beta_array : ndarray
        Array of angles between the LOS and the impact direction at each LOS point (radians).
    theta_matrix : ndarray
        2D array of scattering angles (radians) used in emissivity calculations for each LOS point.
    x_los_cm : ndarray
        Distances along the LOS in centimeters.
    T_HI : float, optional
        Kinetic temperature of neutral hydrogen in Kelvin, used for absorption profile width (default: 1e6 K).

    Returns
    -------
    float
        Integrated Lyman-alpha intensity along the LOS in units of [photons cm⁻² s⁻¹ sr⁻¹].

    Notes
    -----
    - Emissivity is first computed at each LOS point using `calculate_emissivity`.
    - The final intensity is obtained by numerical integration over the LOS distance.
    """

    vw_cms = vw_kms * 1e5
    
    emissivity_values = []
    
    for r, n_e, R_i, beta, theta_arr in zip(r_3d_rsun, n_e_array, Ri_array, beta_array, theta_matrix):
        emissivity = calculate_emissivity(r, vw_cms, n_e, R_i, beta, theta_arr)
        emissivity_values.append(emissivity)
    
    emissivity_values = np.array(emissivity_values)

    # Integrate over line-of-sight distance x_los_cm
    intensity = trapz(emissivity_values, x_los_cm)
    
    return intensity


def integrate_intensity_los_II(vw_profile_kms, r_3d_rsun, n_e_array, Vw_array, Ri_array, beta_array, theta_matrix, x_los_cm, T_HI=1e6):
  
    vw_profile_cms = vw_profile_kms * 1e5
    
    emissivity_values = []

    for r, n_e, Vw, R_i, beta, theta_arr in zip(r_3d_rsun, n_e_array, vw_profile_cms, Ri_array, beta_array, theta_matrix):
        emissivity = calculate_emissivity(r, Vw, n_e, R_i, beta, theta_arr)
        emissivity_values.append(emissivity)
    
    emissivity_values = np.array(emissivity_values)

    # Integrate over line-of-sight distance x_los_cm
    intensity = trapz(emissivity_values, x_los_cm)
    
    return intensity

