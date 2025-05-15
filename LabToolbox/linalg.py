import numpy as np

def comm(A, B):
    """
    Computes the commutator [A, B] = AB - BA.

    Parameters
    ----------
    A : np.ndarray
        The first matrix.
    B : np.ndarray
        The second matrix.

    Returns
    -------
    np.ndarray
        The matrix commutator AB - BA.
    """
    return A @ B - B @ A

def acomm(A, B):
    """
    Computes the anticommutator {A, B} = AB + BA.

    Parameters
    ----------
    A : np.ndarray
        The first matrix.
    B : np.ndarray
        The second matrix.

    Returns
    -------
    np.ndarray
        The matrix anticommutator AB + BA.
    """
    return A @ B + B @ A

def fmatrix(f, A):
    """
    Applies a function to the eigenvalues of a matrix and returns the transformed matrix.

    This function computes the eigenvalues and eigenvectors of the matrix `A`, applies 
    the function `f` to the eigenvalues, and then reconstructs the matrix using the 
    transformed eigenvalues.

    Parameters
    ----------
    f : function
        A function that will be applied to the eigenvalues of the matrix `A`.
    A : numpy.ndarray
        A square matrix for which the eigenvalues and eigenvectors will be computed.

    Returns
    -------
    numpy.ndarray
        The matrix obtained by applying the function `f` to the eigenvalues of `A`.
    """
    eigvals, eigvecs = np.linalg.eig(A)
    D = np.diag(f(eigvals))
    return eigvecs @ D @ np.linalg.inv(eigvecs)

def proj(u, v):
    """
    Projects vector `v` onto vector `u`.

    This function computes the projection of the vector `v` onto the vector `u` using 
    the formula: proj_u(v) = (v ⋅ u / u ⋅ u) * u.

    Parameters
    ----------
    u : numpy.array
        The vector onto which `v` will be projected.
    v : numpy.array
        The vector to be projected.

    Returns
    -------
    numpy.ndarray
        The projection of vector `v` onto vector `u`.
    """

    # u = np.asarray(u)
    # v = np.asarray(v)

    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("Both inputs must be 1D numpy arrays representing vectors.")
    
    return np.vdot(v, u) * u / np.vdot(u, u)

def sign(A):
    """
    Compute the signature of a matrix, which is the number of positive,
    negative, and zero eigenvalues of the matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The input square matrix.

    Returns
    -------
    tuple
        A tuple (n_positive, n_negative, n_zero), where:
        - n_positive is the number of positive eigenvalues,
        - n_negative is the number of negative eigenvalues,
        - n_zero is the number of zero eigenvalues.

    Examples
    --------
    >>> A = np.array([[1, 2], [2, 1]])
    >>> sign(A)
    (2, 0, 0)

    The matrix [[1, 2], [2, 1]] has two positive eigenvalues and no negative or zero eigenvalues.
    """
    eigvals = np.linalg.eigvals(A)
    n_positive = np.sum(eigvals > 0)
    n_negative = np.sum(eigvals < 0)
    n_zero = np.sum(np.isclose(eigvals, 0))
    return (int(n_positive), int(n_negative), int(n_zero))

def rotate(v, theta):
    """
    Rotate a 2D vector by an angle theta in radians.

    Parameters
    ----------
    v : numpy.ndarray
        The vector to be rotated, should be 2D (2,).
    theta : float
        The rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        The rotated vector.

    Example
    -------
    >>> rotate(np.array([1, 0]), np.pi/2)
    array([0., 1.])
    """
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return rotation_matrix @ v

# qui sotto al posto di axis va un vettore qualunque oppure una str.

def rotate3d(v, axis, theta):
    """
    Rotate a 3D vector by an angle theta around a specified axis.

    Parameters
    ----------
    v : numpy.ndarray
        The vector to be rotated, should be 3D (3,).
    axis : str
        The axis of rotation ('x', 'y', or 'z').
    theta : float
        The rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        The rotated vector.

    Example
    -------
    >>> rotate3d(np.array([1, 0, 0]), 'z', np.pi/2)
    array([0., 1., 0.])
    """
    # Rotazione intorno all'asse x
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta)],
                                    [0, np.sin(theta), np.cos(theta)]])
    # Rotazione intorno all'asse y
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
    # Rotazione intorno all'asse z
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return rotation_matrix @ v

def gramschmid(vect, first=0, norm=True):
    """
    Perform Gram-Schmidt orthogonalization on a list of vectors.

    Parameters
    ----------
    vect : list of array-like
        List of input vectors to be orthogonalized.
    first : int, optional
        Index at which to begin orthogonalization (default is 0).
    norm : bool, optional
        Whether to normalize the resulting vectors (default is True).

    Returns
    -------
    list of numpy.ndarray
        Orthonormal (or orthogonal) list of vectors.
    """

    vect = [np.asarray(v, dtype=float) for v in vect]
    u = vect.copy()
    for i in range(first, len(vect)):
        vec = vect[i].copy()
        for j in range(first, i):
            vec -= proj(u[j], vect[i])
        if norm:
            vec = vec / np.linalg.norm(vec)
        u[i] = vec
    return u

def pauli(index):
    """
    Returns the specified Pauli matrix.

    Parameters
    ----------
    index : int or str
        The index of the Pauli matrix. Accepts 1, 2, 3 or 'x', 'y', 'z'.

    Returns
    -------
    np.ndarray
        The corresponding Pauli matrix.
    """
    if index in [1, 'x']:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif index in [2, 'y']:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif index in [3, 'z']:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError("Index must be 1, 2, 3 or 'x', 'y', 'z'.")

def dirac(index):
    """
    Returns the specified Dirac gamma matrix in the Dirac representation.

    Parameters
    ----------
    index : int
        The index of the gamma matrix (0 to 4). Index 0 returns γ⁰, 
        1–3 return γ¹ to γ³, and 5 returns γ⁵.

    Returns
    -------
    np.ndarray
        The corresponding 4×4 Dirac matrix.
    """
    zero = np.zeros((2, 2), dtype=complex)
    identity = np.eye(2, dtype=complex)
    pauli_matrices = [pauli(i+1) for i in range(3)]

    if index == 0:
        return np.block([
            [identity, zero],
            [zero, -identity]
        ])
    elif index in [1, 2, 3]:
        sigma = pauli_matrices[index - 1]
        return np.block([
            [zero, sigma],
            [-sigma, zero]
        ])
    elif index == 5:
        # gamma^5 = i * gamma^0 * gamma^1 * gamma^2 * gamma^3
        gamma = [dirac(i) for i in range(4)]
        return 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    else:
        raise ValueError("Index must be 0–3 for gamma^mu, or 5 for gamma^5.")

def slash(v):
    """
    Compute the Feynman slash notation for a 4-vector v.

    Parameters
    ----------
    v : array-like of length 4
        The 4-vector (v^0, v^1, v^2, v^3).

    Returns
    -------
    numpy.ndarray
        The Dirac matrix γ^μ v_μ, shape (4, 4).

    Raises
    ------
    ValueError
        If the input is not a 4-vector.
    """
    if len(v) != 4:
        raise ValueError("Input must be a 4-vector (length 4).")
    return sum(v[i] * dirac(i) for i in range(4))

class FourVector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=complex).view(cls)
        if obj.shape != (4,):
            raise ValueError("FourVector must be of length 4.")
        return obj

    def slash(self):
        return slash(self)

def changebasis(v, Bnew, Bold = None, matrix = None):
    """
    Change the basis of a vector from one basis to another.

    Parameters
    ----------
    v : numpy.ndarray
        The vector to be transformed (assumed column vector).
    Bnew : list of numpy.ndarray
        The new basis (list of column vectors).
    Bold : list of numpy.ndarray or None, optional
        The old basis. If None, the standard basis is assumed.
    matrix : numpy.ndarray, optional
        The change-of-basis matrix. If not provided, it will be computed.

    Returns
    -------
    numpy.ndarray
        The vector expressed in the new basis.
    """
    v = np.asarray(v).reshape(-1)
    dim = v.shape[0]

    if matrix is None:
        if Bold is None:
            Bold = [np.eye(dim)[:, i] for i in range(dim)]
        matrix = np.linalg.inv(np.column_stack(Bold)) @ np.column_stack(Bnew)

    return matrix @ v

def orthspace(vect):
    """
    Compute an orthogonal basis for the orthogonal complement of the span of given vectors.

    Parameters
    ----------
    vect : list of numpy.ndarray
        A list of vectors (as column vectors) that span a subspace.

    Returns
    -------
    list of numpy.ndarray
        A list of vectors that form the orthogonal complement of the subspace spanned by the input vectors.

    Raises
    ------
    ValueError
        If the input vectors are linearly dependent.
    """
    # Verifica se i vettori sono linearmente indipendenti
    matrix = np.column_stack(vect)
    if np.linalg.matrix_rank(matrix) != len(vect):
        raise ValueError("The input vectors are linearly dependent.")
    
    # Costruisci la matrice che contiene i vettori di input
    A = np.column_stack(vect)

    # Calcola la base ortogonale usando la decomposizione QR
    # L'ortogonale complementare è dato da una base della null space di A^T
    _, _, vh = np.linalg.svd(A.T)
    
    # La base ortogonale è data dalle righe non nulle di vh (SVD di A^T)
    # Questi rappresentano il complementare ortogonale
    rank = np.linalg.matrix_rank(A)
    null_space_basis = vh[rank:]
    
    # Restituisci i vettori come lista di numpy.ndarray
    return [null_space_basis[i] for i in range(null_space_basis.shape[0])]

def areindependent(vect):
    """
    Check if the given list of vectors are linearly independent.

    Parameters
    ----------
    vect : list of numpy.ndarray
        List of column vectors.

    Returns
    -------
    bool
        True if the vectors are linearly independent, otherwise False.
    """
    matrix = np.column_stack(vect)
    if np.linalg.matrix_rank(matrix) == len(vect):
        return True
    else:
        return False

def ismatrix(matrix, type=None):
    """
    Check various properties of the given matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to check.
    type : str or None, optional
        A logical expression combining properties using "and", "or", "not", 
        e.g., "herm and defpos", "unit or orth", or "herm and not defneg".
        If None, returns the list of satisfied properties.

    Returns
    -------
    list of str or bool
        If `type` is None, returns a list of satisfied properties among:
        ["sym", "herm", "antiherm", "orth", "unit", 
         "defpos", "semidefpos", "defneg", "semidefneg", "inv"].
        If `type` is specified, returns a boolean indicating whether 
        the logical condition is satisfied.

    Notes
    -----
    - "sym"        : symmetric
    - "herm"       : Hermitian
    - "antiherm"   : anti-Hermitian
    - "orth"       : orthogonal
    - "unit"       : unitary
    - "defpos"     : positive definite
    - "semidefpos" : positive semi-definite
    - "defneg"     : negative definite
    - "semidefneg" : negative semi-definite
    - "inv"        : invertible

    Examples
    --------
    >>> ismatrix(np.eye(3))
    ['sym', 'herm', 'defpos', 'orth', 'inv']

    >>> ismatrix(np.eye(3), type="herm and inv")
    True

    >>> ismatrix(np.array([[0, 1], [1, 0]]), type="sym and defpos")
    False
    """

    props = []

    n = matrix.shape[0]
    identity = np.eye(n, dtype=matrix.dtype)

    # Proprietà algebriche
    if np.allclose(matrix, matrix.T):
        props.append("sym")
    if np.allclose(matrix, matrix.conj().T):
        props.append("herm")
    if np.allclose(matrix.conj().T, -matrix):
        props.append("antiherm")
    if matrix.shape[0] == matrix.shape[1] and np.allclose(matrix @ matrix.T, identity):
        props.append("orth")
    if matrix.shape[0] == matrix.shape[1] and np.allclose(matrix.conj().T @ matrix, identity):
        props.append("unit")

    # Autovalori
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        if np.all(eigvals > 0):
            props.append("defpos")
        if np.all(eigvals >= 0):
            props.append("semidefpos")
        if np.all(eigvals < 0):
            props.append("defneg")
        if np.all(eigvals <= 0):
            props.append("semidefneg")
    except np.linalg.LinAlgError:
        pass

    # Invertibilità
    try:
        np.linalg.inv(matrix)
        props.append("inv")
    except np.linalg.LinAlgError:
        pass

    if type is None:
        return props

    # Eval expression
    all_props = {
        "sym", "herm", "antiherm", "orth", "unit",
        "defpos", "semidefpos", "defneg", "semidefneg", "inv"
    }
    env = {p: (p in props) for p in all_props}

    # Sicurezza: tokenizzazione e validazione
    allowed = {"and", "or", "not", "(", ")", *all_props}
    tokens = type.replace("(", " ( ").replace(")", " ) ").split()
    if not all(token in allowed for token in tokens):
        raise ValueError(f"Invalid expression: {type}")

    return eval(type, {"__builtins__": None}, env)

def docommute(A, B):
    """
    Check if the commutator of two operators is zero (i.e., they commute).

    Parameters
    ----------
    A : np.ndarray
        The first operator.
    B : np.ndarray
        The second operator.

    Returns
    -------
    bool
        True if the operators commute, otherwise False.
    """
    return np.allclose(comm(A,B), np.zeros_like(A))