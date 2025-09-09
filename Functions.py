
from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
    from scipy.constants import c

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

    import numpy as np

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

    from importlib.machinery import SourceFileLoader
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



from scipy.signal import find_peaks
import numpy as np

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


import numpy as np

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