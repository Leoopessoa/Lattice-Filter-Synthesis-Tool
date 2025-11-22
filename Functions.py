
#[1] K. Jinguji, Y. Hida, M. Kawachi, “Synthesis of coherent two-port lattice-form optical delay-line circuit,” J. Lightwave Technol., vol. 13, no. 1, pp. 73–82, Jan. 1995.
#[2] K. Jinguji, M. Oguma, “Optical half-band filters using lattice-form waveguide circuits,” J. Lightwave Technol., vol. 18, no. 2, pp. 252–259, Feb. 2000.
#[3] J. D. Domenech and J. Capmany, “Optical filter design and analysis,” in Integrated Photonics, UPV Press, Valencia, Spain, 2013, pp. 141–178.


from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, hilbert
from importlib.machinery import SourceFileLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sympy as sp
from sympy import symbols, expand, Poly, factor, I, exp
from numpy.polynomial import Polynomial
import scienceplots
import itertools
import pandas as pd
import math
import warnings
import logging
from IPython.display import display, Latex

z = symbols('z', complex = True); ω = symbols('ω', real = True); π = np.pi

def addONA(process, λstart,λend,np,ports, x, y):

    """
    Parameters:
        process(ModuleType): Lumerical API used.
        λstart(float): Inital wavelength value.
        λstop(float): Final wavelength value.
        np(int): Number of simulation points.
        ports(int): Number of output ports.
        x(int): Ona x Position
        y(int): Ona y Position
    """

    process.addelement('Optical Network Analyzer')
    process.set('input parameter','start and stop')
    process.set({'number of points'     : np,
                 'number of input ports': ports,
                 'start frequency'      : c/λstart, 
                 'stop frequency'       : c/λend,
                 'x position'           : x,
                 'y position'           : y})

def ConnectONA(process, ONA, component, connections):

    """
    Parameters:
        process(ModuleType): Lumerical API used.
        ONA(string): Name of the ONA.
        component(string): Name of the component.
        connections(int): array with connection orders.
    """
    process.connect(ONA,'output', component,'port '+ str(connections[0]))

    for i in range(1,(len(connections))):
        process.connect(ONA,'input '+str(i), component,'port '+str(connections[i]))
        
    
def Connectcomponents(process, component1, connections1, component2, connections2):

    """
    Parameters:
        process(ModuleType): Lumerical API used.
        component1(string): Name of the component 1.
        component2(string): Name of the component 2.
        connections1(int): array with the connection ports of the component 1.
        connections2(int): array with the connection ports of the component 2.
    """
    if (len(connections1)!=len(connections2)):
        print('invalid connection, component 1 has', len(connections1), 'connections and component 2 has', len(connections2), 'connections')
        return 0;

    for i in range(len(connections1)):
        process.connect(component1,'port '+str(connections1[i]), component2,'port '+str(connections2[i]))
        
    
def GenerateHexMZI( process: str, Loss: float, theta_matrix: list[list[float]], Length: float):
    """
    GenerateHexMZI is a function that simulates a hexagonal MZI (Mach-Zehnder processferometer) setup.
    
    Parameters:
    - Loss (float): A floating-point number representing the loss on each MZI.
    - process (ModuleType): Lumerical API used.
    - theta_matrix (list[list[float]]): A 2D list (6x2 matrix) where each row contains 
      two floating-point numbers representing thetaA and thetaB in radians.
    - Length (float): A floating-point number representing the length of the MZI.
    
    Returns:
    None
    """
 
    process.addelement('MZI_HEX')
    process.set('Loss',Loss)
    process.set('Length', Length)

    # Set the phase of the MZIs
    for i in range(6):
        process.set('Theta'+str(i+1)+'_A', theta_matrix[i][0])
        process.set('Theta'+str(i+1)+'_B', theta_matrix[i][1])
        


def polyfind(coeff,linspace,targetValue,targetAxis='x'):
    """
    # polyfind

    Finds a approximate value (x or y) of a polynomial function, 
    given its coefficients and its target value to reference (in x or y axis)

    coeff : n-order polynomial coefficients (coeff[0]*x^(n) + coeff[1]*x^(n-1) + ...), float array

    linspace : array [a,b,c] for numpy.linspace(a,b,c) execution for x axis, float array

    targetValue : Value to be targetted, float

    targetAxis (optional) : defines the axis in which targetValue will be applied, str]

    ## returns 

    returns a float value:
        - If targetAxis is 'x', this function will return the corresponding y axis value 
        - If targetAxis is 'y', this function will return the corresponding x axis value 
    """

    xArray = np.linspace(linspace[0],linspace[1],linspace[2])
    y = 0

    for i in range(len(coeff)):
        y += coeff[i]*xArray**(len(coeff)-i-1)

    if targetAxis == 'x':
        FoundValue = y[np.argmin(np.abs(xArray-targetValue))]
    elif targetAxis == 'y':
        FoundValue = xArray[np.argmin(np.abs(y-targetValue))]
    else:
        raise NameError(f'fitfind(): No target axis named {targetAxis}. Use x or y.')

    return FoundValue


def eyediagramFix(sigIn, Nsamples, SpS, n=3, ptype="fast", plotlabel=None):
    """
    Imported from Opticommpy library
    
    Plot the eye diagram of a modulated signal waveform. (Altered version) 
    
    Adapted for:
    - Support less sample points
    - Returning a matplotlib.figure.Figure() object

    Parameters
    ----------
    sigIn : array-like
        Input signal waveform.
    Nsamples : int
        Number of samples to be plotted.
    SpS : int
        Samples per symbol.
    n : int, optional
        Number of symbol periods. Defaults to 3.
    ptype : str, optional
        Type of eye diagram. Can be 'fast' or 'fancy'. Defaults to 'fast'.
    plotlabel : str, optional
        Label for the plot legend. Defaults to None.
    

    Returns
    -------
    Returns a matplotlib.figure.Figure() object
    """
    sig = sigIn.copy()

    if not plotlabel:
        plotlabel = " "

    if np.iscomplex(sig).any():
        d = 1
        plotlabel_ = f"{plotlabel} [real]" if plotlabel else "[real]"
    else:
        d = 0
        plotlabel_ = plotlabel

    for ind in range(d + 1):
        if ind == 0:
            y = sig[:Nsamples].real
            x = np.arange(0, y.size, 1) % (n * SpS)
        else:
            y = sig[:Nsamples].imag

            plotlabel_ = f"{plotlabel} [imag]" if plotlabel else "[imag]"
        eyefig = plt.figure()
        if ptype == "fancy":
            f = interp1d(np.arange(y.size), y, kind="cubic")

            Nup = 1 * SpS
            tnew = np.arange(y.size) * (1 / Nup)
            y_ = f(tnew)

            taxis = (np.arange(y.size) % (n * SpS * Nup)) * (1 / Nup)
            imRange = np.array(
                [
                    [min(taxis), max(taxis)],
                    [min(y) - 0.1 * np.mean(np.abs(y)), 1.1 * max(y)],
                ]
            )

            H, xedges, yedges = np.histogram2d(taxis, y_, bins=350, range=imRange)

            H = H.T
            H = gaussian_filter(H, sigma=1.0)

            plt.imshow(
                H,
                cmap="turbo",
                origin="lower",
                aspect="auto",
                extent=[0, n, yedges[0], yedges[-1]],
            )

        elif ptype == "fast":
            y[x == n * SpS] = np.nan
            y[x == 0] = np.nan

            plt.plot(x / SpS, y, color="blue", alpha=0.8, label=plotlabel_)
            plt.xlim(min(x / SpS), max(x / SpS))

            if plotlabel is not None:
                plt.legend(loc="upper left")

        plt.xlabel("symbol period (Ts)")
        plt.ylabel("amplitude")
        plt.title(f"eye diagram {plotlabel_}")
        plt.grid(alpha=0.15)
        plt.close()

    return eyefig

def lumapi():
    """
    Create Lumerical API integration
    
    Returns
    -------
    lumapi : Lumerical API
    """
    sys.path.append('../')
    versions = ['v202','v221','v231','v241','v242']

    print('found!')
    for ver in versions:
        try:
            os.add_dll_directory('C:\\Program Files\\Lumerical\\'+ ver +'\\api\\python\\')
            lumapi = SourceFileLoader('lumapi','C:\\Program Files\\Lumerical\\'+ ver +'\\api\\python\\lumapi.py').load_module()
            print(f'version {ver} found!')
            break
        except:
            print(f'version {ver} not found...')
    print('import not found')

    return lumapi


def FSR(x, y, proximity_threshold = 0):
    """
    Calculate the FSR of a transmission array, dynamically determining proximity threshold.

    Parameters
    ----------
    x: float array
        The x array of values (e.g., wavelength).
    y: float array
        The y array of values for which the calculation will be done (e.g., transmission).
    proximity_threshold: float array
        The minimal value of distance between the peaks ()
    
    Returns
    -------
    fsr: float
        Average value of FSR obtained.
    
    Examples
    --------
    Using a sine wave:

    >>>    x = symbols('x')
    >>>    ax = np.linspace(0,4*π,1000)
    >>>    y = np.sin(2*π*ax)
    >>>    FSR(ax,y)

    (np.float64(1.000026490331886),
    [array([ 0.25157899,  1.24531601,  2.25163197,  3.24536899,  4.25168495,
          5.24542197,  6.25173793,  7.24547495,  8.25179091,  9.24552793,
         10.25184389, 11.24558091, 12.25189688])])
    """
    # Normalize the y array to bring all values between 0 and 1
    y = y / y.max()

    # Detect all peaks in the y data
    # The 'height=0.7' parameter ensures we only consider peaks with values >= 70% of the max
    peaks, properties = find_peaks(y, height=0.65)

    final_peaks = []
    group = []

    # Calculate adaptive proximity threshold if not provided
    if proximity_threshold == 0:
        avg_spacing = np.mean(np.diff(x[peaks]))
        proximity_threshold = avg_spacing * 0.5  # Half of the average peak spacing
    
    # Loop through all detected peaks
    for i, peak in enumerate(peaks):
        # If the current peak is close enough to the previous one, add it to the group
        if i == 0 or (peak - peaks[i-1] <= proximity_threshold):
            group.append(peak)
        else:
            # If the current peak is not close, process the previous group
            # Select the central peak of the group
            central_peak = group[len(group) // 2]
            final_peaks.append(central_peak)
            # Start a new group with the current peak
            group = [peak]
    
    # Process the last group if it exists
    if group:
        central_peak = group[len(group) // 2]
        final_peaks.append(central_peak)
    
    # Calculate differences between the selected peaks
    final_peaks = np.array(final_peaks)
    fsr_values = abs(np.diff(x[final_peaks]))
    
    # Return the mean FSR value if there are enough peaks; otherwise, return None
    return (np.mean(fsr_values) if len(fsr_values) > 0 else None), final_peaks

def FWHM(x, y, peak_idx):
    """
    Calculate the Full Width at Half Maximum (FWHM) for a given peak.

    Parameters
    ----------
    x : array-like
        The x array of values (e.g., wavelength, time, etc.).
    y : array-like
        The y array of values corresponding to the curve (e.g., intensity, amplitude, etc.) in dB.
    peak_idx : int
        Index of the peak in the y array for which to calculate the FWHM.

    Returns
    -------
    fwhm : float
        The Full Width at Half Maximum of the peak.
    left_idx : int
        Index of the left intersection point at half maximum.
    right_idx : int
        Index of the right intersection point at half maximum.
    """
    # Find the half maximum value in dB
    half_max = y[peak_idx] -3 

    # Search to the left for the first point below the half maximum
    left_idx = peak_idx-10
    while y[left_idx] > half_max and left_idx < len(y):
        left_idx -= 5

    # Search to the right for the first point below the half maximum
    right_idx = peak_idx+10
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 5


    # Calculate the FWHM
    fwhm = x[right_idx] - x[left_idx]

    return abs(fwhm), left_idx, right_idx

def hamming_window(N, H):
    """
    Hamming window function for CROW synthesis.

    Parameters
    ----------
    N : int.
        Number of resonators.
    H : int.
        Hamming window parameter.

    Returns
    -------
    fwhm : array-like.
        Coupling coefficients for the CROW synthesis.
    
    Example
    -------
    >>> hamming_window(6, 0.2)
    array([0.688996, 0.833333, 0.977671, 0.977671, 0.833333, 0.688996])
    """

    i = np.arange(N)
    return (1 + H *( np.cos(2 * np.pi * (i - 0.5 * (N - 1)) / N))) / (1 + H)

def calculate_a0(roots_list):
    """
    Calculates the a0 parameter from a list of roots.

    Parameters
    ----------
    roots_list : list or np.ndarray
        A list containing the roots (alpha_k).

    Returns
    -------
    float
        The calculated value of the a0 parameter.

    Notes
    -----
    Returns 1.0 if the denominator is close to zero to avoid division errors.

    Examples
    --------
    >>> roots = [0.5, 0.2-0.1j, -0.8]
    >>> calculate_a0(roots)
    0.3651483716701107
    """
    roots_array = np.array(roots_list)
    #prod_minus = np.prod(np.abs(1 - roots_array)**2)
    #prod_plus = np.prod(np.abs(1 + roots_array)**2)
    prod_minus = np.prod((1 - roots_array)**2)
    prod_plus = np.prod((1 + roots_array)**2)
    denominator = prod_minus + prod_plus

    if abs(denominator) < 1e-15:
        return 1.0
    
    return np.sqrt(1 / denominator)

def build_coeffs_from_roots(all_roots, root_indices):
    """
    Builds polynomial coefficients from a list of roots.

    Parameters
    ----------
    all_roots : np.ndarray
        An array containing all available roots.
    root_indices : list or slice
        The indices used to select the desired roots from `all_roots`.

    Returns
    -------
    fz_coeffs : np.ndarray
        The coefficients of the resulting F(z) polynomial.
    a0 : float
        The leading coefficient a0, calculated for F(z).

    Examples
    --------
    >>> roots = np.array([-0.8, -0.5, 0.1, 0.5, 0.9])
    >>> indices = [0, 2, 4]
    >>> coeffs, a0 = build_coeffs_from_roots(roots, indices)
    >>> print(np.round(coeffs, 4))
    [ 0.4472 -0.2236 -0.3578  0.0447]
    >>> print(round(a0, 4))
    0.4472
    """
    # Select roots based on the provided indices
    selected_roots = all_roots[root_indices]
    
    # Calculate the leading coefficient a0
    a0 = calculate_a0(selected_roots)

    # Build the polynomial coefficients from the selected roots
    fz_coeffs = a0 * np.poly(selected_roots)

    return fz_coeffs, a0


def find_roots_and_build_coeffs(ak_coeffs, roots_indices):
    """
    Finds roots of a polynomial A(z) and builds a new polynomial F(z).

    Parameters
    ----------
    ak_coeffs : list or np.ndarray
        Coefficients of the input polynomial A(z).
    roots_indices : list or slice
        Indices of the roots of A(z) to be used for building F(z).

    Returns
    -------
    fz_coeffs : np.ndarray
        The coefficients of the resulting polynomial F(z).
    a0 : float
        The leading coefficient a0, calculated for F(z).

    Examples
    --------
    >>> ak = [-0.03125, 0, 0.28125, 0.5, 0.28125, 0, -0.03125]
    >>> selected_indices = [3, 4, 5]
    >>> coeffs, a0 = find_roots_and_build_coeffs(ak, selected_indices)
    >>> print(np.round(coeffs, 4))
    [ 0.3415  0.5915  0.1585 -0.0915]
    >>> print(round(a0, 4))
    0.3415
    """
    # Find all roots from the input polynomial coefficients
    all_roots = np.roots(ak_coeffs)

    # Delegate the build process to the other function
    fz_coeffs, a0 = build_coeffs_from_roots(all_roots, roots_indices)

    return fz_coeffs, a0

def regression(gz_coef, hz_coef):
    """
    Performs a regression step to extract coefficients for n-1 'ak' and 'bk'.

    This function calculates the coefficients 'ak' and 'bk', and the angles
    'theta' and 'phi' from the input polynomial coefficients G(z) and H(z).

    Parameters
    ----------
    gz_coef : np.ndarray
        Array with the coefficients of the polynomial G(z).
    hz_coef : np.ndarray
        Array with the coefficients of the polynomial H(z).

    Returns
    -------
    ak : np.ndarray
        Array with the newly calculated 'ak' coefficients.
    bk : np.ndarray
        Array with the newly calculated 'bk' coefficients.
    theta : float
        The calculated theta angle.
    phi : float
        The calculated phi angle.

    Notes
    -----
    Coefficients with an absolute value less than 1e-9 are filtered out to
    prevent numerical noise.

    Examples
    --------
    >>> gz = np.array([0.5, 0.8])
    >>> hz = np.array([-0.5, 0.8])
    >>> ak, bk, theta, phi = regression(gz, hz)
    >>> print("ak:", ak, "bk:", bk, "\ntheta:", theta, "phi:", phi)
    ak: [1.13137085+0.j] bk: [-0.70710678+0.j] 
    theta: -0.7853981633974483 phi: -0.0
    """
    theta = -np.arctan(np.abs(hz_coef[-1]) / np.abs(gz_coef[-1]))

    if len(gz_coef) > 1:
        # Angle of the last coefficient
        numerator = gz_coef[-2] * hz_coef[-1] - gz_coef[-1] * hz_coef[-2]
        denominator = gz_coef[-1]**2 + hz_coef[-1]**2
        phi = -round(np.angle(numerator / denominator), 5)

        ak = np.zeros(len(gz_coef) - 1, dtype=complex)
        bk = np.zeros(len(gz_coef) - 1, dtype=complex)

        for i in range(len(gz_coef) - 1):
            term_gz_cos = gz_coef[-i + 1] * np.cos(theta)
            term_hz_sin = hz_coef[-i + 1] * np.sin(theta)
            ak[i] = (term_gz_cos - term_hz_sin) * np.exp(1j * phi / 2)

            term_gz_sin = gz_coef[-i] * np.sin(theta)
            term_hz_cos = hz_coef[-i] * np.cos(theta)
            bk[i] = (term_gz_sin + term_hz_cos) * np.exp(-1j * phi / 2)

        # Filter out coefficients with very small values to avoid noise
        ak = ak[np.abs(ak) > 1e-9]
        bk = bk[np.abs(bk) > 1e-9]

    else:
        ak = np.array([0])
        bk = np.array([0])
        phi = 0

    return ak, bk, theta, phi

def create_mzi_of_order(inter, order, thetas, phis, dlen=22.88e-6, l=50e-6,
                        ng=4.2, neff=2.44, dx=125,
                        simulation_band=[1450e-9, 1650e-9], points=10001, verbose=True):
    """
    Automatically generates and simulates a cascaded Mach-Zehnder Interferometer (MZI)
    of a given order in Lumerical INTERCONNECT.

    Args:
        inter: The Lumerical INTERCONNECT session API object.
        order (int): The order (N) of the MZI filter to be created.
        thetas (list or np.array): A list of N+1 theta values in radians.
        phis (list or np.array): A list of N phi values in radians for the phase shifters.
        dlen (float): Arm length difference. Defaults to 22.88e-6.
        l (float): Base waveguide length. Defaults to 50e-6.
        ng (float): Waveguide group index. Defaults to 4.2.
        neff (float): Waveguide effective index. Defaults to 2.44.
        dx (float): Horizontal spacing between components. Defaults to 125.
        simulation_band (list): [start, end] of the simulation band in meters.
        points (int): Number of points in the ONA simulation.
        verbose (bool, optional): If True, displays the prints. Defaults to True.
    """
    
    # --- Input Validation ---
    if len(thetas) != order + 1:
        raise ValueError(f"The 'thetas' list must contain {order + 1} elements for a {order}-order MZI.")
    if len(phis) != order:
        raise ValueError(f"The 'phis' list must contain {order} elements for a {order}-order MZI.")

    # --- Workspace Cleanup ---
    try:
        inter.switchtolayout()
    except:
        pass
    inter.selectall()
    inter.delete()

    # --- Initial Calculations ---
    k_coeffs = np.sin(np.array(thetas))**2

    # --- Element Creation ---
    if verbose:
        print(f"Starting the creation of a {order}-order MZI...")

    # First Coupler (theta_0)
    inter.addelement('Waveguide Coupler')
    inter.set({
        'name': 'theta0',
        'x position': 100,
        'y position': 500,
        'coupling coefficient 1': k_coeffs[0]
    })
    inter.set('coupling coefficient 2', inter.getnamed('theta0', 'coupling coefficient 1'))

    # Loop to create the intermediate stages
    for i in range(order):
        stage = i + 1
        
        # Upper Waveguide (with length adjustment on the last stage)
        upper_length = l + dlen if (stage == order and order > 1) else l + 2*dlen
        # For order 1, the first and last stage are the same
        if (order == 1): upper_length = l + dlen

        inter.addelement('Straight Waveguide')
        inter.set({
            'name': f'Upper_{stage}',
            'y position': 200,
            'x position': 100 + (2 * i + 1) * dx,
            'length': upper_length,
            'group index 1': ng,
            'effective index 1': neff
        })

        # Lower Waveguide
        inter.addelement('Straight Waveguide')
        inter.set({
            'name': f'Lower_{stage}',
            'y position': 600,
            'x position': 100 + (2 * i + 1) * dx,
            'length': l,
            'group index 1': ng,
            'effective index 1': neff
        })

        # Optical Phase Shifter
        inter.addelement('Optical Phase Shift')
        inter.set({
            'name': f'phi{stage}',
            'y position': 400,
            'x position': 100 + (2 * i + 1.75) * dx,
            'phase shift': phis[i]
        })
        inter.rotateelement(f'phi{stage}')

        # Next Coupler (theta_1, theta_2, ...)
        inter.addelement('Waveguide Coupler')
        inter.set({
            'name': f'theta{stage}',
            'x position': 100 + 2 * stage * dx,
            'y position': 500,
            'coupling coefficient 1': k_coeffs[stage]
        })
        inter.set('coupling coefficient 2', inter.getnamed(f'theta{stage}', 'coupling coefficient 1'))

    inter.addelement("Optical Network Analyzer")
    inter.set('input parameter','start and stop')
    inter.set({ 'number of points'      : points,
                 'number of input ports': 2,
                 'start frequency'      : c/simulation_band[0], 
                 'stop frequency'       : c/simulation_band[-1],
                 'x position'           : -200,
                 'y position'           : 300})

    # Connections
    if verbose:
        print("Connecting elements...")
    
    # Initial ONA Connection
    inter.connect('ONA_1', 'output', 'theta0', 'port 1')

    # Loop Connections
    for i in range(order):
        stage = i + 1
        inter.connect(f'theta{i}', 'port 3', f'Upper_{stage}', 'port 1')
        inter.connect(f'theta{i}', 'port 4', f'Lower_{stage}', 'port 1')
        
        inter.connect(f'Upper_{stage}', 'port 2', f'phi{stage}', 'port 1')
        inter.connect(f'phi{stage}', 'port 2', f'theta{stage}', 'port 1')

        inter.connect(f'Lower_{stage}', 'port 2', f'theta{stage}', 'port 2')

    # Final ONA Connections
    inter.connect('ONA_1', 'input 1', f'theta{order}', 'port 3')
    inter.connect('ONA_1', 'input 2', f'theta{order}', 'port 4')

def lumapi():
    """
    Create Lumerical API integration
    
    Returns
    -------
    lumapi : Lumerical API
    """
    sys.path.append('../')
    versions = ['v202','v221','v231','v241','v242']

    print('found!')
    for ver in versions:
        try:
            os.add_dll_directory('C:\\Program Files\\Lumerical\\'+ ver +'\\api\\python\\')
            lumapi = SourceFileLoader('lumapi','C:\\Program Files\\Lumerical\\'+ ver +'\\api\\python\\lumapi.py').load_module()
            print(f'version {ver} found!')
            break
        except:
            print(f'version {ver} not found...')
    print('import not found')

    return lumapi

def find_delay(arr1, arr2):
    """
    Finds the delay between two 1D signals using cross-correlation.

    Parameters
    ----------
    arr1 : np.ndarray
        The reference 1D array.
    arr2 : np.ndarray
        The 1D array to be aligned with 'arr1'.

    Returns
    -------
    delay : int
        Calculated delay in number of samples.
    arr1 : np.ndarray
        The original first array.
    arr2_aligned : np.ndarray
        The version of 'arr2' shifted to align with 'arr1'.

    Examples
    --------
    >>> arr1 = np.array([0, 0, 1, 2, 1, 0, 0])
    >>> arr2 = np.array([0, 1, 2, 1, 0, 0, 0])
    >>> delay, _, arr2_aligned = find_delay(arr1, arr2)
    >>> delay
    1
    >>> arr2_aligned
    array([0, 0, 1, 2, 1, 0, 0])
    """
    arr1_centered = arr1 - np.mean(arr1)
    arr2_centered = arr2 - np.mean(arr2)

    correlation = np.correlate(arr1_centered, arr2_centered, mode='full')
    delay = correlation.argmax() - (len(arr2) - 1)
    arr2_aligned = np.roll(arr2, delay)

    return delay, arr1, arr2_aligned

def Maximally_flat_weights(N):
    '''
    Obtain the wieght coefficients for the Maximally flat digital filter order "N"
    
    Parameters
    ----------
    N: int
        Order of the digital filter
    
    Returns
    -------
    W: np.ndarray
        Weight coefficients array
    
    Examples
    --------
    >>> W = Maximally_flat_weights(1)  
    >>> print(W)
    [ 0.5      0.28125  0.      -0.03125]

    References
    ----------
    .. [4] GUMACOS, Constantine. "Weighting Coefficients for Certain Maximally
           Flat Nonrecursive Digital Filters". IEEE TRANSACTIONS ON CIRCUITS
           AND SYSTEMS, VOL. CAS-25, NO. 4, pp. 234-235, Abril de 1978.
    '''
    # Initialization
    W = np.zeros(2*(N+1))
    Ak = np.zeros(N+1)
    W[0] = 1/2 # W_0 definition
    
    for k in range(N+1): # Create Ak,N array
            Ak[k]= ((-1)**k/(2*k+1))*(math.factorial(N)*math.factorial(N+1))/(math.factorial(N-k)*math.factorial(N+k+1))
    W1 = 1/(4*np.sum(Ak)) # Normalization factor W1
    
    for i in range(N+1): # Create W array
        W[2*i+1] = W1*Ak[i]
    return W

def create_symmetric_filter(W):
    """
    Creates a symmetric filter (impulse response) from the W coefficients.

    Parameters
    ----------
    W : np.ndarray
        The input weight coefficients from maximally_flat_weights(),
        where W[0] is the center tap.

    Returns
    -------
    np.ndarray
        The full, symmetric filter impulse response.

    Example
    -------
    >>> W = [0.5, 0.2, -0.3]
    >>> A_k = create_symmetric_filter(W)
    >>> print(A_k)
    [-0.3  0.2  0.5  0.2 -0.3]
    """
    # The center tap of the filter is W[0].
    center_tap = 0.5
    # Create symmetric left and right sides.
    right_side = W[1:]
    left_side = right_side[::-1]

    # Concatenate the matrix with the middle value
    symmetric_filter = np.concatenate((left_side, [center_tap], right_side))
    
    return symmetric_filter

def calculate_delta_L(fsr: float, ng: float, lambda_0: float) -> float:
    """
    Calculates the optical path difference (delta L) based on FSR, group refractive index,
    and central wavelength.

    Args:
        fsr (float): Free Spectral Range in meters (m) or the same unit as lambda_0.

        ng (float): Group refractive index (dimensionless).

        lambda_0 (float): Central wavelength in meters (m) or the same unit as FSR.

    Returns:
        float: The calculated optical path difference (Delta L) in meters (m).

    Example:
    >>> fsr_val_m_example = 25e-9 # FSR of 25 nm
    >>> ng_val_example = 4.2
    >>> lambda_0_val_m_example = 1550e-9 # Central wavelength 1550 nm
    >>> calculated_delta_L_m = calculate_delta_L(fsr_val_m_example, ng_val_example, lambda_0_val_m_example)
    >>> print(f"Calculated ΔL: {calculated_delta_L_m * 1e6:.4f}µm ")
    Calculated ΔL: 22.8810µm 
    """

    # Calculate Delta L using the formula
    delta_L = (lambda_0**2) / (ng * fsr)
    
    return delta_L

def filter_with_indexes(N = 2, idx = [0,1,2], type='maximally_flat', fsr=25e-9, ng=4.2, lambda_0=1550e-9):
    """
    Filters the coefficients of a polynomial based on specified indices.

    Parameters
    ----------
    N : int
        The order of the polynomial filter. Default is 2.
    idx : list
        A list of indices to select from the polynomial coefficients.
        Default is [0, 1, 2].
    type : str
        The type of filter to create. Default is 'maximally_flat'.
    fsr : float
        The free spectral range of the desired filter.
    ng : float
        The group index of the circuit.
    lambda_0 : float
        The central wavelength of the circuit.

    Returns
    -------
    theta: np.ndarray
        The coupling angle of the MZIs.
    phi: np.ndarray
        The phase shift angle of the MZIs.
    a: np.ndarray
        The coefficients of the polynomial G(z) based on the specified indices.
    b: np.ndarray
        The coefficients of the polynomial H(z) based on the specified indices.

    Examples
    --------
    >>> Initial definitions 
    >>> lambda_0=1550e-9
    >>> ng = 4.2 # Silicon waveguide group index
    >>> desired_order = 2
    >>> theta, phi, a, b = filter_with_indexes(N=2, idx=[3, 4, 5], type='maximally_flat', fsr=0)
    >>> print(f"| Theta2: {theta[0]/np.pi:.4f}π, Phi2: {phi[0]:.4f} | \n"
    Order	Theta (π)	   k	  Phi
    Theta2	-0.2500π	0.85355	-3.1416
    Theta1	-0.3333π	0.75002	-0.0000
    Theta0	-0.4167π	0.62938	--------
    The ΔL for the desired FSR is: 22.8810 μm
    """
    # Initialize coefficients
    if type != 'maximally_flat':
        raise ValueError("Currently, only 'maximally_flat' type is supported.")
    elif type == 'maximally_flat':
        # Generate the maximally flat weights and create the symmetric filter
        W = Maximally_flat_weights(N-1)

    # Obtain the symmetric filter coefficients
    A_k = create_symmetric_filter(W)
    B_k = create_symmetric_filter(-W)
    a, a0 = find_roots_and_build_coeffs(roots_indices=idx, ak_coeffs=A_k)
    Gz = -a  # Negate G(z) coefficients to match the formulation
    Hz, b0 = find_roots_and_build_coeffs(roots_indices=idx, ak_coeffs=B_k)  # Assuming B_k is similar to A_k for this example
    
    # MZI filter coefficients theta and phi calculations
    theta = np.zeros(N+1)
    phi = np.zeros(N+1)
    a = Gz; b = Hz;
    for i in range(N+1):
        a, b, theta[i], phi[i] = regression(gz_coef=a, hz_coef=b)

    # Lenghth of the filter based on the free spectral range
    if fsr > 0:
        delta_L = calculate_delta_L(fsr, ng, lambda_0)
    else:
        delta_L = 0
    
    # Filter coefficients based on the provided indices
    return theta, phi, Gz, Hz, delta_L

def find_filter_pairs(N, filter_type='maximally_flat', a0_precision=4, imag_threshold=1e-3, match_tolerance=1e-3, hilbert_tol=1.0, verbose=True):
    """
    Generates and analyzes a pair of primary filters (A_k, B_k) based on N and
    filter_type, then finds and pairs their derived filters (G(z), H(z)).

    Parameters
    ----------
    N : int
        The desired filter order.
    filter_type : str, optional
        The type of filter to generate. Currently, only 'maximally_flat'
        is supported. Defaults to 'maximally_flat'.
    a0_precision : int, optional
        The decimal precision for comparing a0 values to find uniques. Defaults to 4.
    imag_threshold : float, optional
        The tolerance for acceptable imaginary parts in coefficients. Defaults to 1e-3.
    match_tolerance : float, optional
        The tolerance for pairing a0 values between G(z) and H(z). Defaults to 1e-3.
    hilbert_tol : float, optional
        The tolerance for the Hilbert transform test. Defaults to 1.0 + 1e-6.
    verbose : bool, optional
        If True, prints progress updates during execution. Defaults to True.

    Returns
    -------
    list
        A list of pairs. Each pair is a list containing two numpy arrays:
        [G(z) coefficients, H(z) coefficients].
        Returns an empty list if no pairs are found.

    Examples
    --------
    >>> # Define the desired filter order
    >>> N_input = 1
    >>> # Call the main function to get the paired coefficients
    >>> paired_filters = find_filter_pairs(N=N_input, verbose=False)
    >>> print(f"\nPair #{i+1}")
    >>> print(f"G(z) Coeffs (ak): {np.round(pair[0], 5)}")
    >>> print(f"H(z) Coeffs (bk): {np.round(pair[1], 5)}")
    Pair #1
    G(z) Coeffs (ak): [-0.5 -0.5]
    H(z) Coeffs (bk): [ 0.5 -0.5]
    """
    
    # Internal Setup & Coefficient Generation
    if filter_type != 'maximally_flat':
        raise ValueError("Currently, only 'maximally_flat' type is supported.")
    
    if verbose:
        print(f"--- Generating filters for N={N} and type='{filter_type}' ---")
        
    expected_len = 2 * N
    W = Maximally_flat_weights(N - 1)
    A_k = create_symmetric_filter(W)
    B_k = create_symmetric_filter(-W)

    # Root Calculation
    roots_A = np.sort_complex(np.roots(A_k))
    combos_A = list(itertools.combinations(range(len(roots_A)), 2 * N - 1))
    roots_B = np.sort_complex(np.roots(B_k))
    combos_B = list(itertools.combinations(range(len(roots_B)), 2 * N - 1))

    # Processing A_k to find G(z) candidates
    if verbose:
        print(f"\n--- Processing A_k to find unique G(z) filters ---")
        print(f"Testing {len(combos_A)} root combinations for A_k...")
        
    results_G = []
    seen_a0s_A = set()
    for indices in combos_A:
        try:
            idx_list = list(indices)
            coeffs, a0_val = build_coeffs_from_roots(roots_A, root_indices=idx_list) # Build coefficients from selected roots
            a0_round = np.round(a0_val, a0_precision) # Round a0 to the specified precision
            if a0_round in seen_a0s_A: continue # Skip if this a0 has been seen before
            final_coeffs = -coeffs # Negate coefficients to match the G(z) formulation
            if len(final_coeffs) != expected_len: continue # Ensure the length matches the expected polynomial degree
            if np.any(np.abs(np.imag(final_coeffs)) > imag_threshold): continue # Check for significant imaginary parts
            seen_a0s_A.add(a0_round) # Add the rounded a0 to the seen set
            if not passes_hilbert_test(np.real(final_coeffs), tol=hilbert_tol):
                continue
            if not passes_paley_wiener_test(np.real(final_coeffs)):
                continue
            results_G.append({'a0': a0_val, 'coeffs': final_coeffs}) # Store the result
        except Exception:
            continue
    if verbose:
        print(f"Found {len(results_G)} unique G(z) candidates.")

    # Processing B_k to find H(z) candidates
    if verbose:
        print(f"\n--- Processing B_k to find unique H(z) filters ---")
        print(f"Testing {len(combos_B)} root combinations for B_k...")
        
    results_H = []
    seen_a0s_B = set()
    for indices in combos_B:
        try:
            idx_list = list(indices)
            final_coeffs, a0_val = build_coeffs_from_roots(roots_B, root_indices=idx_list) # Build coefficients from selected roots
            a0_round = np.round(a0_val, a0_precision) # Round a0 to the specified precision
            if a0_round in seen_a0s_B: continue # Skip if this a0 has been seen before
            if len(final_coeffs) != expected_len: continue # Ensure the length matches the expected polynomial degree
            if np.any(np.abs(np.imag(final_coeffs)) > imag_threshold): continue # Check for significant imaginary parts
            seen_a0s_B.add(a0_round) # Add the rounded a0 to the seen set
            if not passes_hilbert_test(np.real(final_coeffs), tol=hilbert_tol):
                continue
            if not passes_paley_wiener_test(np.real(final_coeffs)):
                continue
            results_H.append({'a0': a0_val, 'coeffs': final_coeffs}) # Store the result
        except Exception:
            continue
    if verbose:
        print(f"Found {len(results_H)} unique H(z) candidates.")

    # Pairing G(z) and H(z) filters
    if verbose:
        print("\n--- Pairing G(z) and H(z) filters based on closest a0 value ---")

    matched_coeffs_list = []
    available_H = list(results_H)
    
    for g_result in results_G:
        best_match_h = None
        min_diff = float('inf')

        for h_result in available_H:
            diff = abs(g_result['a0'] - h_result['a0'])
            if diff < min_diff:
                min_diff = diff
                best_match_h = h_result
        
        if best_match_h and min_diff < match_tolerance:
            g_coeffs = np.real(g_result['coeffs'])
            h_coeffs = np.real(best_match_h['coeffs'])
            matched_coeffs_list.append([g_coeffs, h_coeffs])
            available_H.remove(best_match_h)
            
    if verbose:
        print(f"Found {len(matched_coeffs_list)} matched pairs.")
        
    return matched_coeffs_list

def regression_for_filter_pairs(N, paired_filters):
    """
    Performs regression on the paired filter coefficients to calculate theta and phi values.

    Parameters
    ----------
    N : int
        The order of the filters.
    paired_filters : list
        A list of pairs, where each pair contains G(z) and H(z) coefficients.

    Returns
    -------
    list
        A list of tuples containing theta and phi values for each pair.
    
    Examples
    --------    
    >>> N_input = 2
    >>> f_results = pd.DataFrame()
    >>> print("\n--- Regression Results for Filter Pairs ---")
    >>> paired_filters = find_filter_pairs(N=N_input, verbose=False)
    >>> regression_results = regression_for_filter_pairs(N_input, paired_filters)
    >>> for i, (theta, phi) in enumerate(regression_results):
    >>>     df_results = thetas_phis_df(theta, phi, True)
    --- Regression Results for Filter Pairs ---
    Order	Theta (π)	k	Phi
    Theta2	-0.2500π	0.85357	-0.0000
    Theta1	-0.3976π	0.65811	-3.1416
    Theta0	-0.2852π	0.81235	--------
    ...
    """
    results = []
    
    for pair in paired_filters:
        a = pair[0]
        b = pair[1]
        theta = np.zeros(N+1)
        phi = np.zeros(N+1)
        
        for i in range(N+1):
            a, b, theta[i], phi[i] = regression(gz_coef=a, hz_coef=b)
        
        results.append((theta, phi))
    
    return results

def thetas_phis_df(theta, phi, verbose=True):
    """
    Generates a pandas DataFrame summarizing the theta, k and phi values.

    Parameters
    ----------
    theta (list or np.array): 
        A list of theta values in radians for the MZI couplers.
    phi (list or np.array):
        A list of phi values in radians for the MZI phase shifters.
    verbose (bool, optional):
        If True, displays the DataFrame. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the order, theta in π, k values, and phi values.
    
    Examples
    --------
    >>> theta = [np.pi/4, np.pi/3, np.pi/2]
    >>> phi = [0, np.pi/2, np.pi]
    >>> df = thetas_phis_df(theta, phi, verbose=True)
    Order	Theta (π)	k	Phi
    Theta2	0.2500π	0.85355	0.0000
    Theta1	0.3333π	0.75000	1.5708
    Theta0	0.5000π	0.50000	--------
    """
    N = len(theta) - 1
    data = []

    for i in range(N, -1, -1):
        k_val = round(np.sin(theta[N-i])**2, 5)  # get k value
        if i == 0:
            data.append([f"Theta{i}", f"{theta[N-i]/np.pi:.4f}π", k_val, "--------"])
        else:
            data.append([f"Theta{i}", f"{theta[N-i]/np.pi:.4f}π", k_val, f"{phi[N-i]:.4f}"])

    df = pd.DataFrame(data, columns=["Order", "Theta (π)", "k", "Phi"])
    if verbose:
        display(df.style.hide(axis="index").format(precision=5))
    return df

def passes_hilbert_test(coeffs, tol=1.0):
    """
    Applies the Hilbert transform to a sequence of coefficients and checks
    whether the envelope magnitude stays below the specified tolerance.

    Parameters
    ----------
    coeffs : list or np.ndarray
        Sequence of real filter coefficients (e.g., from G(z) or H(z)).
    tol : float, optional
        Maximum allowed envelope magnitude. Defaults to 1.0 + 1e-6.

    Returns
    -------
    bool
        True if all points in the Hilbert envelope are <= tol.
        False otherwise or if the computation fails.

    Examples
    --------
    >>> coeffs = [0.5, -0.3, 0.1, -0.05]
    >>> passes_hilbert_test(coeffs)
    True
    >>> coeffs = [10, -10, 5, -5]
    >>> passes_hilbert_test(coeffs)
    False
    """
    try:
        analytic = hilbert(coeffs)
        envelope = np.abs(analytic)
        return np.all(envelope < tol)
    except Exception:
        return False

def passes_paley_wiener_test(coeffs, num_points=2048):
    """
    Checks the Paley-Wiener condition for a discrete-time filter polynomial.
    
    The condition requires that the integral of |ln|H(e^jw)|| over the unit
    circle is finite. This ensures realizability for stable, causal filters.

    Parameters
    ----------
    coeffs : list or np.ndarray
        Filter coefficients (polynomial sequence).
    num_points : int, optional
        Number of frequency samples used to approximate the integral.
        Defaults to 2048.

    Returns
    -------
    bool
        True if the integral is finite (passes Paley-Wiener test).
        False otherwise.

    Examples
    --------
    >>> coeffs = [0.5, -0.3, 0.1]
    >>> passes_paley_wiener_test(coeffs)
    True
    """
    try:
        coeffs = np.asarray(coeffs, dtype=float)
        # Evaluate frequency response on the unit circle
        w = np.linspace(-np.pi, np.pi, num_points)
        ejw = np.exp(-1j * np.outer(w, np.arange(len(coeffs))))
        H = ejw @ coeffs
        mag = np.abs(H)
        mag = np.clip(mag, 1e-12, None)  # avoid log(0)

        # Approximate the integral
        integral_val = np.trapezoid(np.abs(np.log(mag)), w)
        return np.isfinite(integral_val)
    except Exception:
        return False

def lambidify_filter(N=2, verbose=True, points=10001, fsr=25e-9, simulationband=[1450e-9, 1650e-9]):
    """
    Creates the lambidified version of the filter coefficients for a given order N.

    Parameters
    ----------
    N : int
        The order of the filter. Default is 2.
    verbose : bool, optional
        If True, displays the prints. Defaults to True.
    points : int, optional
        Number of frequency points for evaluation. Default is 10001.
    fsr : float
        The free spectral range of the desired filter.
    simulationband : list
        The simulation band [start, end] in meters.

    Returns
    -------
    A: np.ndarray
        The lambidified coefficients of the polynomial G(z).
    B: np.ndarray
        The lambidified coefficients of the polynomial H(z).

    Examples
    --------
    >>> N_input = 2
    >>> A, B = lambidify_filter(N=N_input, verbose=False)
    >>> print(A,B)
    [1. 1. 1. ... 1. 1. 1.] [9.99200722e-16 1.08197558e-15 1.07838537e-15 ... 1.34224734e-15
     1.30582249e-15 9.99200722e-16]
    """
    W = Maximally_flat_weights(N)
    A_coeffs = create_symmetric_filter(W)
    B_coeffs = create_symmetric_filter(-W)  # complementary pair

    #Define A(z), B(z)
    A_z = sum(coeff * z**(-i) for i, coeff in enumerate(A_coeffs))
    B_z = sum(coeff * z**(-i) for i, coeff in enumerate(B_coeffs))

    #Z to frequency substitution
    A_freq = A_z.subs(z, sp.exp(sp.I*ω))
    B_freq = B_z.subs(z, sp.exp(sp.I*ω))

    #Lambdify for numerical evaluation
    A_func = sp.lambdify(ω, A_freq, 'numpy')
    B_func = sp.lambdify(ω, B_freq, 'numpy')

    #Frequency grid (ω in [-π, π])
    # Obtaining the values for omega_vals
    desired_fsr = 25e-9
    pi_limits = (simulationband[0]-simulationband[-1])/fsr*π # Calculate the omega span
    omega_vals = np.linspace(-pi_limits, pi_limits, points)

    #Evaluate magnitude responses
    A_vals = np.abs(A_func(omega_vals))
    B_vals = np.abs(B_func(omega_vals))

    if verbose:
        # Plot results directly in ω-domain
        plt.figure(figsize=(8,5))
        plt.plot(omega_vals, 20*np.log10(A_vals), label="|A(ω)| (dB)")
        plt.plot(omega_vals, 20*np.log10(B_vals), label="|B(ω)| (dB)")
        plt.xlim(-2*np.pi, 2*np.pi)
        plt.ylim(-100, 5)
        plt.xlabel("Frequency ω (rad)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"Maximally Flat Symmetric Filters (N={N})")
        plt.xticks([-2*np.pi,-1*np.pi, 0, 1*np.pi, 2*np.pi],
            [r"$-2π$", r"$-π$", "0", r"$π$", r"$2π$"])
        plt.legend()
        plt.grid(True)
        plt.show()
    return A_vals, B_vals

def curve_correlation(ref_curve, test_curve, threshold=0.7):
    """
    Compare two curves by Pearson correlation (no phase shift allowed).

    Parameters
    ----------
    ref_curve : array_like
        Reference curve (e.g. symmetric maximally flat filter).
    test_curve : array_like
        Curve to be tested (e.g. regression output).
    threshold : float, optional
        Minimum acceptable correlation for considering the curves "similar".
        Default is 0.7.

    Returns
    -------
    corr_value : float
        Pearson correlation coefficient in [-1, 1].
    is_similar : bool
        True if corr_value >= threshold, False otherwise.

    Examples
    --------
    >>> corr, ok = curve_correlation([1,2,3],[1.1,2.0,3.05], threshold=0.7)
    >>> print(corr, ok)
    0.999015263178192 True
    """
    ref = np.asarray(ref_curve, dtype=float)
    tst = np.asarray(test_curve, dtype=float)

    if len(ref) != len(tst):
        raise ValueError("Curves must have the same length for correlation test.")

    corr_value = abs(np.corrcoef(ref, tst)[0, 1])
    return corr_value, corr_value >= threshold