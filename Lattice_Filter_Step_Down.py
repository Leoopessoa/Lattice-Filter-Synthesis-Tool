from sympy import expand, Poly, collect
from numpy import angle, sqrt
from cmath import exp as e

def Lattice_Filter_Step_Down(An , Bn, phi, z):
    """
    Compute the step down recursion obtaining the coupling and phase parameters
    See "Optical Filter Design and Analysis: A Signal Processing Approach Christi K. Madsen, Jian H. Zhao"

    Parameters
    ----------
    An, Bn : sympy.core.add.Add
        Transfer function polynomial terms for the N-th order 
        of the MZI filter
    phi : float
        Phase difference between the MZI arms of the N-th order iteration
    z: sympy.core.symbol.Symbol
        Sympy variable symbol
    
    Returns
    -------
    A, B : sympy.core.add.Add
        Transfer function polynomial terms for the (N-1)-th order 
        of the MZI filter
    coefficients : float
        Coupling coefficients for the MZI coupler, beeing k, c and s, respectively

    Examples
    --------
    Using a simple zero phase second order equation

    >>> Lattice_Filter_Step_Down(z**(-2) + 2*z**(-1),0.75*z**(-2) + z**(-1), 0 ,z)
    (2.2 + 1.25/z, -0.4/z, [0.36, 0.8, 0.6])
    """

    # Obtaining the last coefficient of A and B
    An_Poly = Poly(An)
    An_coefs = An_Poly.all_coeffs()
    ann = float(An_coefs[0])

    Bn_Poly = Poly(Bn)
    Bn_coefs = Bn_Poly.all_coeffs()
    bnn = float(Bn_coefs[0])

    # Using ann and bnn to obtain the coupling coefficient

    k = float(abs(bnn/ann)**2/((1)+abs(bnn/ann)**2))
    c = float(sqrt(1-k))
    s = float(sqrt(k))
    coefficients = [k, c, s] # Creating the coefficients array
    
    # Step down for B
    B = expand((-s*An+c*Bn))
    B_Poly = Poly(B)
    B_coefs = B_Poly.all_coeffs()

    print(B_coefs)

    if len(B_coefs) == len(Bn_coefs):
        # Remove the highest order term z^-n
        B_coefs.pop(0)
        # Recreate the expression B without the highest order term
        B = sum(coef * (1/z)**i for i, coef in enumerate(B_coefs))
        # Recreate the polynomial B with the correct order
        B = sum(coef * z**(-i) for i, coef in enumerate(B_coefs))


    # Calculating A~
    Atilde = expand((c*An+s*Bn))
    Atilde_Poly = Poly(Atilde)
    Atilde_coefs = Atilde_Poly.all_coeffs()

    # Calculating A
    A = expand(Atilde*z*e(1j*phi))
    A_Poly = Poly(A, 1/z)
    A_coefs = A_Poly.all_coeffs()
    
    
    # Debugging in case the polinome reaches A0 and B0
    if((len(Bn_coefs)==2)):
        B = B_coefs[0]
        A = A_coefs[0]

 # Check if there is a term with z for A
    collected_terms = collect(A, z, evaluate=False)
    if z in collected_terms:
        # Divide all terms by z
        A = A / z
        # Recalculate the coefficients after division
        A_Poly = Poly(A, 1/z)
        A_coefs = A_Poly.all_coeffs()
        # Remove the highest order term z^-n
        A_coefs.pop(0)
        # Recreate the expression A without the highest order term
        A = sum(coef * (1/z)**i for i, coef in enumerate(A_coefs))

    return A, B, coefficients #, phase


def FSR(x,y):
    """
    Calculate the FSR of a transmission array

    Parameters
    ----------
    x: float array
        The x array of values (wavelength).
    y: float array
        The y array of values that the calculation will be done (transmission).
    
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

    np.float64(1.000026490331886)
    """
    y = y/y.max() # Normalize the array
    y[y<= 0.5] = 0 # Remove lower peaks that could interfere with the FSR calculations
    peaks, _ = find_peaks(y) 
    fsr_values = np.diff(x[peaks]) # Calculate the diference between peaks
    return mean(sum(fsr_values)/len(fsr_values)) # Mean value of the FSR

