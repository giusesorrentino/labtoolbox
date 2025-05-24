import numpy as np
from ._helper import format_stokes

"""
Module for polarization optics calculations using Jones and Stokes/Mueller calculi.

Provides functions for:
- Jones vectors and matrices to model fully polarized light interactions with optical elements such as polarizers, waveplates, and phase retarders.
- Stokes vectors to represent polarization states, including partially polarized and unpolarized light.
- Mueller matrices to describe transformations of Stokes vectors by optical elements.
- Conversions between Jones and Stokes/Mueller frameworks for interoperability.
- Rotation operations for both Jones and Mueller matrices to handle arbitrary orientations.

Designed for optics applications, this module supports analysis of polarized light in systems involving linear and circular polarizers, waveplates, and general birefringent materials.
"""

# --- Jones Calculus Functions ---

def jvec(axis):
    """
    Returns the normalized Jones vector for a specified polarization type.

    Parameters
    ----------
        axis : str
            Polarization type ('h' for horizontal, 'v' for vertical, 'd' for +45°, 
            'a' for -45°, 'r' for right circular, 'l' for left circular).

    Returns
    -------
        np.ndarray: 2x1 array representing the Jones vector.

    Raises
    ------
        ValueError: If the type is not recognized.
    """
    if axis not in ["h", "v", "d", "a", "r", "l"]:
        raise ValueError("Invalid polarization type. Use 'h', 'v', 'd', 'a', 'r', or 'l'.")
    elif axis == "h":
        return np.array([[1], [0]], dtype=complex)
    elif axis == "v":
        return np.array([[0], [1]], dtype=complex)
    elif axis == "d":
        return np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
    elif axis == "a":
        return np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)
    elif axis == "r":
        return np.array([[1/np.sqrt(2)], [-1j/np.sqrt(2)]], dtype=complex)
    else:
        return np.array([[1/np.sqrt(2)], [1j/np.sqrt(2)]], dtype=complex)

def linpol(axis = None, angle=None):
    """
    Returns the Jones matrix for a linear polarizer.

    This function returns the 2x2 Jones matrix of a linear polarizer either aligned along
    a canonical axis ('h', 'v', 'd', 'a') or along a generic axis specified by an angle.

    Parameters
    ----------
    axis : str or None, optional
        Polarizer type:
            - 'h' : horizontal (transmits x-polarized light),
            - 'v' : vertical (transmits y-polarized light),
            - 'd' : +45° (diagonal, transmits at +π/4 from x-axis),
            - 'a' : -45° (anti-diagonal, transmits at -π/4 from x-axis).
        If `angle` is specified, this parameter is ignored.
    
    angle : float or None, optional
        Angle θ in radians specifying the transmission axis of the polarizer
        with respect to the horizontal (x-axis). A positive angle corresponds
        to a counterclockwise (anticlockwise) rotation from the x-axis.
        If provided, `type` is ignored.

    Returns
    -------
    numpy.ndarray
        2x2 complex numpy array representing the Jones matrix of the polarizer.
    """

    if angle is not None:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c**2, c * s], [c * s, s**2]], dtype=complex)
    
    if axis is None:
        raise ValueError("You must provide either 'type' or 'angle'.")

    if axis == "h":
        return np.array([[1, 0], [0, 0]], dtype=complex)
    elif axis == "v":
        return np.array([[0, 0], [0, 1]], dtype=complex)
    elif axis == "d":
        return 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    elif axis == "a":
        return 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
    else:
        raise ValueError("Invalid 'axis'. Must be one of: 'h', 'v', 'd', 'a'.")

def circpol(axis):
    """
    Returns the Jones matrix for a circular polarizer.

    Parameters
    ----------
        axis : str
            Polarizer type ('r' for right circular, 'l' for left circular).

    Returns
    -------
        np.ndarray: 2x2 array representing the Jones matrix.

    Raises
    ------
        ValueError: If the type is not recognized.
    """
    if axis not in ["r", "l"]:
        raise ValueError("Invalid polarization type. Use 'r', or 'l'.")
    if axis == "r":
        return 0.5 * np.array([[1, +1j], [-1j, 1]], dtype=complex)
    else:
        return 0.5 * np.array([[1, -1j], [+1j, 1]], dtype=complex)

def waveplate(kind='general', theta=0.0, eta=None, axis=None):
    """
    Returns the Jones matrix of a birefringent waveplate (linear retarder).

    Parameters
    ----------
    kind : str, optional
        Type of waveplate:
            - 'quarter' : Quarter-wave plate (η = π/2),
            - 'half'    : Half-wave plate    (η = π),
            - 'general' : Arbitrary phase delay η (must be provided).
        Default is 'general'.

    theta : float, optional
        Orientation of the fast axis in radians, measured counterclockwise 
        from the horizontal (x-axis). Ignored if 'axis' is specified as 'h' or 'v'.
        Default is 0.

    eta : float or None, optional
        Phase delay (in radians) between fast and slow axes.
        Required if kind = 'general'.

    axis : str or None, optional
        Shortcut for horizontal or vertical axis:
            - 'h' : horizontal (θ = 0)
            - 'v' : vertical   (θ = π/2)
        Overrides `theta` if provided.

    Returns
    -------
    np.ndarray
        2x2 complex Jones matrix representing the waveplate.
    """
    if axis is not None:
        if axis == 'h':
            theta = 0.0
        elif axis == 'v':
            theta = np.pi / 2
        else:
            raise ValueError("Invalid axis value. Use 'h', 'v', or None.")

    if kind == 'quarter':
        eta = np.pi / 2
    elif kind == 'half':
        eta = np.pi
    elif kind == 'general':
        if eta is None:
            raise ValueError("For 'general' waveplate, specify eta explicitly.")
    else:
        raise ValueError("Invalid kind. Use 'quarter', 'half', or 'general'.")

    c = np.cos(theta)
    s = np.sin(theta)
    eieta = np.exp(1j * eta)

    return np.exp(-1j * eta / 2) * np.array([
        [c**2 + eieta * s**2, (1 - eieta) * c * s],
        [(1 - eieta) * c * s, s**2 + eieta * c**2]
    ], dtype=complex)

def ephase(eta, theta, phi):
    """
    Returns the Jones matrix for an arbitrary birefringent material (elliptical phase retarder).

    Parameters
    ----------
        eta : float
            Relative phase retardation between fast and slow axes (in radians).
        theta : float
            Angle of the fast axis with respect to the horizontal (in radians).
        phi : float
            Circularity parameter (in radians, between -pi/2 and pi/2).

    Returns
    -------
        np.ndarray: 2x2 array representing the Jones matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.exp(-1j * eta / 2) * np.array([
        [c**2 + np.exp(1j * eta) * s**2, (1 - np.exp(1j * eta)) * np.exp(-1j * phi) * c * s],
        [(1 - np.exp(1j * eta)) * np.exp(1j * phi) * c * s, s**2 + np.exp(1j * eta) * c**2]
    ], dtype=complex)

def rmatrix(theta):
    """
    Returns the 2D rotation matrix used to rotate Jones vectors or matrices.

    Parameters
    ----------
    theta : float
        Rotation angle in radians. Positive values correspond to 
        counterclockwise rotation of the polarization ellipse in the x-y plane.

    Returns
    -------
    np.ndarray
        2x2 real-valued rotation matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=complex)

def rotate(matrix, theta):
    """
    Rotates a Jones matrix by angle theta about the optical axis.

    Parameters
    ----------
    matrix : np.ndarray
        2x2 Jones matrix representing the optical element.
    theta : float
        Rotation angle in radians. Positive values correspond to 
        counterclockwise rotation of the optical element as viewed 
        along the direction of light propagation.

    Returns
    -------
    np.ndarray
        2x2 Jones matrix representing the rotated optical element.

    Notes
    -----
    The rotated matrix is given by:
        J_rot = R(-θ) @ J @ R(θ)
    where R(θ) is the real-valued 2D rotation matrix.
    """
    R = rmatrix(theta)
    R_inv = rmatrix(-theta)
    return R_inv @ matrix @ R

def apply(jones_vector, jones_matrix):
    """
    Applies a Jones matrix to a Jones vector (polarized light).

    Parameters
    ----------
    jones_vector : np.ndarray
        2-element complex array representing the input Jones vector.
    jones_matrix : np.ndarray
        2x2 complex array representing the Jones matrix of the optical element.

    Returns
    -------
    np.ndarray
        2-element complex array representing the output Jones vector.
    """
    return jones_matrix @ jones_vector

def intensity(vector):
    """
    Calculates the intensity of a Jones vector (proportional to the sum of squares of absolute values).

    Parameters
    ----------
        vector : np.ndarray
            2x1 Jones vector.

    Returns
    -------
        float: Intensity of the light.
    """
    return np.abs(vector[0, 0])**2 + np.abs(vector[1, 0])**2

# --- Stokes Vector Functions ---

def svec(axis):
    """
    Returns the Stokes vector for a specified polarization type.

    Parameters
    ----------
        axis : str
            Polarization type ('h' for horizontal, 'v' for vertical, 'd' for +45°, 
            'a' for -45°, 'r' for right circular, 'l' for left circular, 'un' for unpolarized).

    Returns
    -------
        np.ndarray: 4x1 array representing the Stokes vector.
    """
    if axis not in ["h", "v", "d", "a", "r", "l", "un"]:
        raise ValueError("Invalid polarization type. Use 'h', 'v', 'd', 'a', 'r', or 'l'.")
    elif axis == "h":
        return np.array([[1], [1], [0], [0]], dtype=float)
    elif axis == "v":
        return np.array([[1], [-1], [0], [0]], dtype=float)
    elif axis == "d":
        return np.array([[1], [0], [1], [0]], dtype=float)
    elif axis == "a":
        return np.array([[1], [0], [-1], [0]], dtype=float)
    elif axis == "r":
        return np.array([[1], [0], [0], [-1]], dtype=float)
    elif axis == "l":
        return np.array([[1], [0], [0], [1]], dtype=float)
    else:
        return np.array([[1], [0], [0], [0]], dtype=float)

def JtoS(jones_vector):
    """
    Converts a Jones vector to a Stokes vector.

    Parameters
    ----------
        jones_vector : np.ndarray
            2x1 Jones vector representing the polarization state.

    Returns
    -------
        np.ndarray: 4x1 Stokes vector [S0, S1, S2, S3].
    """
    Ex, Ey = jones_vector[0, 0], jones_vector[1, 0]
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))
    return np.array([[S0], [S1], [S2], [S3]], dtype=float)

def poldeg(stokes_vector):
    """
    Calculates the degree of polarization from a Stokes vector.

    Parameters
    ----------
        stokes_vector : np.ndarray
            4x1 Stokes vector.

    Returns
    -------
        float: Degree of polarization (between 0 and 1).
    """
    S0 = stokes_vector[0, 0]
    if S0 == 0:
        return 0.0
    S1, S2, S3 = stokes_vector[1, 0], stokes_vector[2, 0], stokes_vector[3, 0]
    polarized_intensity = np.sqrt(S1**2 + S2**2 + S3**2)
    return polarized_intensity / S0

# --- Mueller Matrix Functions ---

def mpol(axis=None, angle=None):
    """
    Returns the Mueller matrix for a polarizer.

    Parameters
    ----------
    axis : str or None, optional
        Polarizer type:
            - 'h' : horizontal (transmits x-polarized light),
            - 'v' : vertical (transmits y-polarized light),
            - 'd' : +45° (diagonal, transmits at +π/4 from x-axis),
            - 'a' : -45° (anti-diagonal, transmits at -π/4 from x-axis),
            - 'r' : right circular,
            - 'l' : left circular.
        If `angle` is specified, this parameter is ignored.

    angle : float or None, optional
        Angle θ in radians specifying the transmission axis of the polarizer
        with respect to the horizontal (x-axis). A positive angle corresponds
        to a counterclockwise (anticlockwise) rotation from the x-axis.
        If provided, `axis` is ignored.

    Returns
    -------
    np.ndarray
        4×4 Mueller matrix of the polarizer.

    Raises
    ------
    ValueError
        If neither a valid axis nor an angle is specified.
    """
    c2 = np.cos(2 * angle)
    s2 = np.sin(2 * angle)

    _mpol_angle = 0.5 * np.array([
        [1,     c2,      s2,     0],
        [c2,  c2**2,  c2 * s2,   0],
        [s2,  c2 * s2, s2**2,    0],
        [0,     0,       0,      0]
    ], dtype=float)

    if angle is not None:
        return _mpol_angle(angle)

    if axis == "h":
        return _mpol_angle(0)
    elif axis == "v":
        return _mpol_angle(np.pi / 2)
    elif axis == "d":
        return _mpol_angle(np.pi / 4)
    elif axis == "a":
        return _mpol_angle(-np.pi / 4)
    elif axis == "r":
        return 0.5 * np.array([
            [1,  0,  0, -1],
            [0,  0,  0,  0],
            [0,  0,  0,  0],
            [-1, 0,  0,  1]
        ], dtype=float)
    elif axis == "l":
        return 0.5 * np.array([
            [1,  0,  0, 1],
            [0,  0,  0, 0],
            [0,  0,  0, 0],
            [1,  0,  0, 1]
        ], dtype=float)
    else:
        raise ValueError("Specify a valid axis ('h', 'v', 'd', 'a', 'r', 'l') or an angle in radians.")

def mwaveplate(kind='general', theta=0.0, eta=None, axis=None):
    """
    Returns the Mueller matrix of a birefringent waveplate (linear retarder).

    Parameters
    ----------
    kind : str, optional
        Type of waveplate:
            - 'quarter' : Quarter-wave plate (η = π/2),
            - 'half'    : Half-wave plate    (η = π),
            - 'general' : Arbitrary phase delay η (must be provided).
        Default is 'general'.

    theta : float, optional
        Orientation of the fast axis in radians, measured counterclockwise 
        from the horizontal (x-axis). Ignored if `axis` is specified.
        Default is 0.

    eta : float or None, optional
        Phase delay (in radians) between fast and slow axes.
        Required if kind = 'general'.

    axis : str or None, optional
        Shortcut for horizontal or vertical axis:
            - 'h' : horizontal (θ = 0)
            - 'v' : vertical   (θ = π/2)
        Overrides `theta` if provided.

    Returns
    -------
    np.ndarray
        4×4 Mueller matrix representing the waveplate.
    """
    if axis is not None:
        if axis == 'h':
            theta = 0.0
        elif axis == 'v':
            theta = np.pi / 2
        else:
            raise ValueError("Invalid axis. Use 'h', 'v', or None.")

    if kind == 'quarter':
        eta = np.pi / 2
    elif kind == 'half':
        eta = np.pi
    elif kind == 'general':
        if eta is None:
            raise ValueError("For 'general' waveplate, you must specify eta.")
    else:
        raise ValueError("Invalid kind. Use 'quarter', 'half', or 'general'.")

    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    ce = np.cos(eta)
    se = np.sin(eta)

    return np.array([
        [1, 0, 0, 0],
        [0, c2**2 + s2**2 * ce, c2 * s2 * (1 - ce), -s2 * se],
        [0, c2 * s2 * (1 - ce), s2**2 + c2**2 * ce,  c2 * se],
        [0, s2 * se,           -c2 * se,            ce]
    ], dtype=float)

def mrmatrix(theta):
    """
    Returns the 4x4 Mueller rotation matrix for a coordinate rotation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians. Positive values correspond to 
        counterclockwise rotation of the polarization ellipse in the x-y plane.

    Returns
    -------
    np.ndarray
        4x4 real-valued Mueller rotation matrix.
    """
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return np.array([
        [1,  0,   0,  0],
        [0,  c2,  s2, 0],
        [0, -s2,  c2, 0],
        [0,  0,   0,  1]
    ], dtype=float)


def mrotate(matrix, theta):
    """
    Rotates a Mueller matrix by angle theta about the optical axis.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 real-valued Mueller matrix representing the optical element.
    theta : float
        Rotation angle in radians. Positive values correspond to 
        counterclockwise rotation of the optical element as viewed 
        along the direction of light propagation.

    Returns
    -------
    np.ndarray
        4x4 real-valued Mueller matrix representing the rotated optical element.

    Notes
    -----
    The rotated matrix is given by:
        M_rot = R(-θ) @ M @ R(θ)
    where R(θ) is the 4x4 Mueller rotation matrix.
    """
    R = mrmatrix(theta)
    R_inv = mrmatrix(-theta)
    return R_inv @ matrix @ R


def mapply(stokes_vector, mueller_matrix):
    """
    Applies a Mueller matrix to a Stokes vector.

    Parameters
    ----------
    stokes_vector : np.ndarray
        4-element real array representing the input Stokes vector.
    mueller_matrix : np.ndarray
        4x4 real-valued Mueller matrix representing the optical element.

    Returns
    -------
    np.ndarray
        4-element real array representing the output Stokes vector.
    """
    return mueller_matrix @ stokes_vector

# --- Conversion Between Jones and Mueller ---

def JtoM(jones_matrix):
    """
    Converts a Jones matrix to a Mueller matrix.

    Parameters
    ----------
        jones_matrix : np.ndarray
            2x2 Jones matrix.

    Returns
    -------
        np.ndarray: 4x4 Mueller matrix.
    """
    # Pauli matrices
    sigma = [
        np.array([[1, 0], [0, 1]], dtype=complex),  # sigma_0
        np.array([[1, 0], [0, -1]], dtype=complex), # sigma_1
        np.array([[0, 1], [1, 0]], dtype=complex),  # sigma_2
        np.array([[0, -1j], [1j, 0]], dtype=complex)  # sigma_3
    ]
    
    mueller = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            # Compute the Kronecker product and trace
            A = np.kron(jones_matrix, np.conj(jones_matrix))
            B = np.kron(sigma[i], sigma[j])
            mueller[i, j] = 0.5 * np.real(np.trace(A @ B))
    
    return mueller

# --- Polarizzation State ---

def polstate(vector):
    """
    Analyze the polarization state from a Jones or Stokes vector.

    Parameters
    ----------
    vector : array-like
        Input polarization state. Can be either:
        - Jones vector: complex array with 2 elements [Ex, Ey]
        - Stokes vector: real array with 4 elements [S0, S1, S2, S3]

    Returns
    -------
    P : float
        Degree of polarization
    I : float
        Total intensity (S0)
    Q : float
        Linear polarization along x-axis (S1/S0)
    U : float
        Linear polarization at 45° (S2/S0)
    V : float
        Circular polarization (S3/S0)
    """
    vector = np.array(vector)

    # Calcola i parametri di Stokes
    if vector.size == 2:
        # Jones vector
        Ex, Ey = vector[0], vector[1]

        # Compute Stokes parameters
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = np.abs(Ex)**2 - np.abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = -2 * np.imag(Ex * np.conj(Ey))
    elif vector.size == 4:
        # Stokes vector
        S0, S1, S2, S3 = vector
    else:
        raise ValueError("Input must be a 2-element Jones vector or a 4-element Stokes vector.")

    # Normalizza i parametri di Stokes
    if S0 == 0:
        raise ValueError("Total intensity (S0) cannot be zero.")
    s1 = S1 / S0
    s2 = S2 / S0
    s3 = S3 / S0

    # Calcola il grado di polarizzazione (P)
    P = np.sqrt(S1**2 + S2**2 + S3**2)/S0
    if P > 1.0:
        P = 1.0  # Normalizza se supera 1 (errore numerico)

    # Calcola gli angoli ψ (orientamento) e χ (ellitticità)
    psi = 0.5 * np.arctan2(S2, S1)  # Angolo di orientamento in radianti

    denominator = np.sqrt(S1**2 + S2**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = 0.5 * np.arctan(np.divide(S3, denominator, where=denominator!=0))
        chi = np.where(denominator == 0,
                    0.5 * np.pi * np.sign(S3),  # Quando denominator è zero
                    chi)

    tolerance = 1e-2

    if np.isclose(P, 0.0, atol=tolerance):
        polarization_type = "Unpolarized"
        handedness = "None"
    elif np.isclose(s3, 0.0, atol=tolerance):  # Linear
        polarization_type = "Linear"
        handedness = "None"
    elif np.isclose(abs(s3), 1.0, atol=tolerance):  # Circular
        polarization_type = "Circular"
        handedness = "Right-Handed (RH)" if s3 > 0 else "Left-Handed (LH)"
    else:  # Elliptical
        polarization_type = "Elliptical"
        handedness = "Right-Handed (RH)" if s3 > 0 else "Left-Handed (LH)"

    # Formatta i valori
    I_str = format_stokes(S0, is_percentage=False)
    Q_str = format_stokes(s1, is_percentage=True)
    U_str = format_stokes(s2, is_percentage=True)
    V_str = format_stokes(s3, is_percentage=True)
    P_str = format_stokes(P, is_percentage=True)
    psi_str = format_stokes(np.degrees(psi), is_percentage=False)
    chi_str = format_stokes(np.degrees(chi), is_percentage=False)

    # Prepara le stringhe dei risultati
    lines = [
        f"I (Intensity)               {I_str}",
        f"Q (Linear Polarization 0°)  {Q_str}",
        f"U (Linear Polarization 45°) {U_str}",
        f"V (Circular Polarization)   {V_str}",
        "-"*0,  # placeholder per la linea divisoria
        f"Degree of Polarization      {P_str}",
        f"Orientation Angle (ψ)       {psi_str} degrees",
        f"Ellipticity Angle (χ)       {chi_str} degrees",
        f"Polarization Handedness     : {handedness}",
        f"Polarization Type           : {polarization_type}"
    ]

    # Calcola la lunghezza massima delle linee (ignorando il placeholder)
    max_length = max(len(line) for line in lines if len(line) > 0)
    # Aggiorna la linea divisoria per uniformità
    divider = "-" * max_length

    # Titolo e decorazioni dinamiche
    title = "Polarization Analysis Results"
    border = "=" * max_length
    centered_title = title.center(max_length)

    # Stampa dinamica e ben formattata
    print(border)
    print(centered_title)
    print(border)
    for line in lines:
        if len(line) == 0:  # sostituisce il placeholder con la linea divisoria dinamica
            print(divider)
        else:
            print(line)
    print(border)

    # Restituisci i parametri
    return P, S0, s1, s2, s3