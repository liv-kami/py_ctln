# ─────────────────────────── Libraries ────────────────────────────

import numpy as np
from itertools import combinations, permutations, chain
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import pickle
from importlib.resources import files
from importlib.metadata import version
import requests
from pathlib import Path
import multiprocessing as mp
import math
import sympy as sp
from sympy import symbols, Matrix, zeros, latex, factor

# ────────────────────── Known Networks Class ──────────────────────

class _KnownNetworks:
    """A helper class for managing requests to use existing lists of
    known classes of CTLNs. This class allows us to provide these lists
    without forcing the user to load them all ahead of time, as these
    lists can get quite large.

    Note that this class is not designed to be used *directly*,
    but rather a user should access these lists through the CTLN class
    via CTLN.collections.method_name_here()

    Methods
    -------
    _mat_to_d6(sA)
        Not intended for use by the end user. This handles the
        conversion of an adjacency matrix to a d6 byte string.
    _mats_to_d6(cls, sAlist, save_to)
        Not intended for use by the end user. This handles converting a
        list of adjacency matrices to d6 byte strings.
    _d6_to_mat(ds)
        Not intended for use by the end user. This handles converting a
        d6 byte string to an adjacency matrix.
    _read_d6_file(cls, path)
        Not intended for use by the end user. This handles reading a
        list of adjacency matrices from a d6 file.
    _load_data(path_ref)
        Not intended for use by the end user. This handles the loading of
        the pkl files and returns the list for the user
    _convert_mat_to_pkl(mat_ref, save_name, mat_part)
        Not intended for use by the end user. This handles the
        conversion of old .mat files to the python-preferred .pkl format.
    all_n(n)
        Returns a list of all CTLNs with n nodes.
    core_n(n)
        Returns a list of all CTLNs with n nodes that are *core motifs*.
    strongly_core_n(n)
        Returns a list of all CTLNs with n nodes that are *strongly core
        motifs*.
    """

    @staticmethod
    def _mat_to_d6(sA):
        """A method to convert an adjacency matrix to a d6 byte string.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix as a numpy array.

        Returns
        -------
            The byte string of the adjacecny matrix in d6 format.
        """
        # get size of matrix and create bytes
        n = sA.shape[0]
        res = bytearray(b'&')

        # Encodes N(n) for size of graph
        if n <= 62:
            res.append(n + 63)
        elif n <= 258047:
            res.append(126)
            res.append((n >> 12) + 63)
            res.append(((n >> 6) & 63) + 63)
            res.append((n & 63) + 63)
        else:
            res.extend([126, 126])
            res.append((n >> 30) + 63)
            res.append(((n >> 24) & 63) + 63)
            res.append(((n >> 18) & 63) + 63)
            res.append(((n >> 12) & 63) + 63)
            res.append(((n >> 6) & 63) + 63)
            res.append((n & 63) + 63)

        # Get bits from matrix
        bits = []
        for row in sA:
            bits.extend(row)

        # Pad with zeros to make multiple of 6 as required by d6 format
        while len(bits) % 6 != 0:
            bits.append(0)

        # Convert to 6-bit integers and offset by 63 as required by format
        for i in range(0, len(bits), 6):
            chunk = bits[i:i + 6]
            val = 0
            for bit in chunk:
                val = (val << 1) | bit
            res.append(val + 63)

        # return the results
        return bytes(res)

    @classmethod
    def _mats_to_d6(cls, sAlist:list, save_to:str):
        """A method that converts a list of adjacency matrices to a d6
        file.

        Parameters
        ----------
        sAlist : array-like
            A list of adjacency matrices.
        save_to : string
            A path to save the d6 file to.
        """
        with open(save_to, 'wb') as f:
            for sA in sAlist:
                f.write(cls._mat_to_d6(sA) + b'\n')

    @staticmethod
    def _d6_to_mat(ds:bytes):
        """A function for reading an .d6 file.

        Parameters
        ----------
        ds : string
            A string of the .d6 file in the proper format

        Returns
        -------
        matrix : array-like
            The adjacency matrix from the .d6 string.
        """

        # Check it starts with & as it should
        if not ds.startswith(b'&'):
            raise ValueError(
                "Invalid d6 format: string must start with '&'")

        # get rid of the &
        s = [b - 63 for b in ds[1:]]

        # Handles the N(n) part of the d6 format
        if s[0] < 63:
            n = s[0]
            s = s[1:]
        elif s[1] < 63:
            n = (s[1] << 12) + (s[2] << 6) + s[3]
            s = s[4:]
        else:
            n = (s[2] << 30) + (s[3] << 24) + (s[4] << 18) + \
                (s[5] << 12) + (s[6] << 6) + s[7]
            s = s[8:]

        # Handles getting the bits for the adj mat.
        bits = []
        for val in s:
            for i in range(5, -1, -1):
                bits.append((val >> i) & 1)

        # Construct the resulting adjacency matrix and return it
        return [bits[i*n : (i+1)*n] for i in range(n)]

    @classmethod
    def _read_d6_file(cls, path):
        """A method that reads an .d6 file and converts it to a
        adjacency matrices.

        Parameters
        ----------
        path
            The file path to the .d6 file.

        Returns
        -------
        collection : array-like
            A list of adjacency matrices.
        """
        collection = []

        if isinstance(path, str):
            with open(path, 'rb') as f:
                for line in f:
                    if not line: continue
                    collection.append(cls._d6_to_mat(line.strip()))
            return collection
        else:
            with path.open('rb') as f:
                for line in f:
                    if not line: continue
                    collection.append(cls._d6_to_mat(line.strip()))
            return collection

    @classmethod
    def _read_d6_file_generator(cls, path):
        """An alternate method for _read_d6_file that works as a
        generator, useful for larger files and parallelization.

        Parameters
        ----------
        path
            The file path to the .d6 file.

        Returns
        -------
        Yields adjacency matrices.
        """
        if isinstance(path, str):
            with open(path, 'rb') as f:
                for line in f:
                    if not line: continue
                    yield cls._d6_to_mat(line.strip())
        else:
            with path.open('rb') as f:
                for line in f:
                    if not line: continue
                    yield cls._d6_to_mat(line.strip())

    @staticmethod
    def _load_data(path_ref):
        """A method to load a list of CTLNs from a pkl data file.

        Parameters
        ----------
        path_ref : string
            The path to the pkl data file. Should be provided by
            importlib.files function.

        Returns
        -------
        Returns the requested list of CTLNs.
        """
        with path_ref.open('rb') as f:
            return pickle.load(f)

    @staticmethod
    def _convert_mat_to_pkl(mat_path: str, save_name: str, mat_part: str =
    'sAcell'):
        """Converts a given existing .mat file to a pkl file for
        implementing as a known network in the package. Saves the
        resulting pkl file to the data folder to be included in the
        subsequent release. Not intended for use beyond being helpful
        for the maintenance and development of this package.

        When we had all of our code in MatLab, our lists of different
        CTLNs were stored in .mat files which are pretty inconvenient to
        use in python, especially if we want to have them included with a
        package like we do here. So, this method assists in converting
        those old files into the pkl format for including in the package.

        Parameters
        ----------
        mat_path : string
            The file path to where the .mat file is currently stored.
        save_name : string
            The name of the file to save the pkl file to. Do *not*
            include the file extension .pkl, it is added automatically.
        mat_part : string
            The name of the "part" of the mat file to be read as the
            list of matrices. This is typically 'sAcell' for the files I
            have seen us use, but the option to change it is given here
            in case that is not always true. (Defaults to 'sAcell')
        """

        # Imports the necessary function for loading .mat files in
        # python from scipy
        from scipy.io import loadmat

        # Grabs the list of matrices from the .mat file and converts it
        # into the format we need for python to use them efficiently
        mats = list(loadmat(mat_path).get(mat_part).flatten())

        # Saves the newly formatted list to a .pkl file in the data
        # folder to be included in the package distribution.
        with open(f'known_network_data/{save_name}.pkl', 'wb') as f:
            pickle.dump(mats, f)

    @classmethod
    def all_n(cls, n):
        """A method for obtaining a list of all CTLNs with n nodes.

        Parameters
        ----------
        n : integer
            The number of nodes to obtain all CTLNs for.

        Returns
        -------
        Returns the requested list.

        Raises
        ------
        ValueError
            If the requested list cannot be found.
        """
        if n < 5:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"all_{n}.pkl")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: all_n({n})')
            return cls._load_data(path_ref)
        else:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"all_{n}.d6")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: all_n({n})')
            return cls._read_d6_file(path_ref)

    @classmethod
    def core_n(cls, n):
        """A method for obtaining a list of all CTLNs with n nodes that
        are *core motifs*.

        Parameters
        ----------
        n : integer
            The number of nodes to obtain all core CTLNs for.

        Returns
        -------
        Returns the requested list.
        """
        if n<5:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"core_{n}.pkl")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: core_n({n})')
            return cls._load_data(path_ref)
        else:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"core_{n}.d6")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: core_n({n})')
            return cls._read_d6_file(path_ref)

    @classmethod
    def strongly_core_n(cls,n):
        """A method for obtaining a list of all CTLNs with n nodes that
        are *strongly core motifs*.

        Parameters
        ----------
        n : integer
            The number of nodes to obtain all core CTLNs for.

        Returns
        -------
        Returns the requested list.
        """
        if n<5:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"strongly_core_{n}.pkl")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: strongly_core_n({n})')
            return cls._load_data(path_ref)
        else:
            path_ref: Path = files("py_ctln.known_network_data") / (
                f"strongly_core_{n}.d6")
            if not path_ref.exists():
                raise ValueError(f'Sorry, we do not yet have the list you '
                                 f'requested: strongly_core_n({n})')
            return cls._read_d6_file(path_ref)

# ──────────────────────── Main Ctln Funcs ─────────────────────────

class CTLN:
    """A class used to provide functions for Combinatorial Threshold
    Linear Network (CTLN) calculations and research.

    ...

    Attributes
    ----------
    epsilon : float, optional
        The value to use for the epsilon parameter (default is 0.51).
    delta : float, optional
        The value to use for the delta parameter (default is 1.76).
    collections : _KnownNetworks
        A pointer for accessing collections of known CTLNs.

    Methods
    -------
    _get_graph_colors(n)
        A method to create colors for graphing a CTLN
    _check_adjacency(sA)
        A method to check the validity of an adjacency matrix
        to prevent errors.
    set_params(epsilon, delta)
        Allows the user to define the values for the parameters epsilon
        and delta
    get_w_mat(sA)
        Creates the W matrix from the adjacency matrix.
    check_fp(sA, sig)
        Checks if a given subgraph (sigma) is a fixed point support
        of a given CTLN
    check_stability(sA, sig)
        Checks whether a given subgraph (sigma) is a stable fixed point
        or not (unstable)
    get_fp(sA)
        A method that finds all of the fixed points, their supports,
        and their stability for a given CTLN.
    threshlin_ode(sA,t,x0)
        A method for solving the system of piecewise linear ordinary
        differential equations to get the firing rates of the neurons
        over time for a given set of initial conditions.
    get_soln(sA)
        A method for obtaining the solution for a CTLN
    plot_graph(sA, ax,show)
        A method that plots the graph of the CTLN.
    plot_soln(sA)
        A method that plots both the graph and the solution of the CTLN.
    run_ctln_model_script(sA)
        An alias for plot_soln.
    is_uid(sA)
        A method for determining if a CTLN is uniform in-degree.
    is_uod(sA)
        A method for determining if a CTLN is uniform out-degree.
    is_core(sA)
        A method for determining if a CTLN is a core motif.
    is_permitted(sA)
        A method for determining if a CTLN is a permitted motif.
    find_graphical_domination(sA,types_to_look_for)
        A method for finding graphical domination relationships within a
        CTLN
    is_strongly_connected(sA)
        A method for determining if a CTLN is strongly connected.
    is_weakly_connected(sA)
        A method for determining if a CTLN is weakly connected.
    is_connected(sA)
        A method for determining if a CTLN is connected (weakly or
        strongly).
    is_strongly_core(sA)
        A method for determining if a CTLN is *strongly* core motif.
    is_hamiltonian(sA)
        A method for determining if a CTLN is hamiltonian (Contains a
        hamiltonian cycle of size n)
    get_projection_direction(sA)
        A method for getting the projection direction to use for
        plotting the projection of the solution onto two dimensions.
    plot_projection(sA, dim1, dim2, ax, show)
        A method for plotting the projection of the solution onto two
        dimensions.
    plot_grayscale(soln, ax)
        A method for plotting the solution as a grayscale heatmap over time.
    parallel_run(matrices, method, num_processes)
        A method that applies a CTLN class method to multiple matrices
        in parallel using multiprocessing for efficient batch processing.
    _check_for_updates()
        A method that checks if there is an update available for the package 
        and alerts the user if there is. Called on import.
    is_circulant(sA)
        A method for determining if a CTLN is circulant, and if so, what the
        ordering and forward edges are.
    is_clique_union(sA, tau_partition)
        A method for determining if a CTLN is a clique union with respect
        to a particular partitioning of the nodes.
    is_directional(sA, omega, tau)
        A method for determining if a CTLN is directional with respect to a
        particular omega and tau partitioning of the graph.
    get_circulant_graph(n, forward_edges)
        A method for generating the adjacency matrix for a circulant graph
        with n nodes and the specified forward edges.
    _make_all_embeddings(n)
        A method for producing all of the possible embeddings to test when checking parameter dependence
    _check_survival_param_ind(cls, sA, sigma,k_of_interest, dom)
        A method for checking if a particular sigma survives the addition of particular dominator nodes k.
    _check_survival_all_embeddings(cls, sA_sig, sigma)
        A method for checking the survival of a particular sigma with respect to all possible domination embeddings.
    """

    epsilon: float = 0.51
    delta: float = 1.76

    collections = _KnownNetworks()

    @staticmethod
    def _get_graph_colors(n):
        """A method to create colors for graphing a CTLN

        Parameters
        ----------
        n : integer
            The number of nodes to be graphed.

        Returns
        -------
        colors : array-like
            List of hex codes for each color.
        """

        # Defines the colors to be used for the first few neurons in the graph based on previously used defaults
        our_colors = ['#CC0000', '#CC8C00', '#808080', '#269900', '#0080B3']

        # If our defaults are enough, use them
        if n <= 5:
            return our_colors[:n]
        
        # Otherwise generate more colors as needed
        else:
            # Creates n colors evenlt distributed along the rainbow
            colors = plt.get_cmap('rainbow', n-5)
            colors = colors(np.linspace(0, 1, n-5))

            # Converts the colors to hex codes
            colors = [mcolors.to_hex(color) for color in colors]

            # Set the first few colors to match the colors used previously 
            colors.insert(0,'#0080B3')
            colors.insert(0,'#269900')
            colors.insert(0,'#808080')
            colors.insert(0,'#CC8C00')
            colors.insert(0,'#CC0000')

            # Returns the list of hex codes
            return colors

    @staticmethod
    def _check_adjacency(sA):
        """Checks the validity of an adjacency matrix to prevent errors.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix to be checked.

        Raises
        ------
        ValueError
            If the given matrix is not a valid adjacency matrix.

        Returns
        -------
        sA : array-like
            The checked and converted adjacency matrix.
        """

        # Convert the adjacency matrix into a numpy array for faster
        # calculations and additional functionality.
        if not isinstance(sA, np.ndarray):
            try:
                sA = np.array(sA)
            except:
                # If numpy fails to understand the input matrix due to
                # formatting issues, let the user know
                raise ValueError(
                    'The given adjacency matrix was formatted '
                    'incorrectly. Please try again.')

        # Check that the matrix is 2 dimensional
        if sA.ndim != 2:
            raise ValueError(
                'The given adjacency matrix must be 2 dimensional. '
                'Please try again.')

        # Check that the matrix is a square matrix
        if sA.shape[0] != sA.shape[1]:
            raise ValueError(
                'The given adjacency matrix must be a square matrix. '
                'Please try again.')

        # Check that the matrix has 0s along the diagonal
        if not np.all(np.diag(sA) == 0):
            raise ValueError(
                'The given adjacency matrix must have 0s on the '
                'diagonal. Please try again.')

        # Check that the matrix is binary (all entries are 1 or 0)
        if not np.all(np.isin(sA, [0, 1])):
            raise ValueError(
                'All entries in the given adjacency matrix must be 1 or '
                '0. Please try again.')

        # Return the converted adjacency matrix once determined to be valid
        return sA

    @classmethod
    def set_params(cls, epsilon: float = 0.51, delta: float = 1.76):
        """Allows the user to define the values for the parameters
        epsilon and delta

        Parameters
        ----------
        epsilon : float, optional
            The value to use for the epsilon parameter (default is 0.51).
        delta : float, optional
            The value to use for the delta parameter (default is 1.76).
        """

        # Checks that Delta and Epsilon are in their legal ranges
        if not delta > 0 : raise ValueError('Delta must be greater than 0')
        if not (0 < epsilon < (delta / (delta + 1))):
            raise ValueError('Epsilon must be positive and less than ('
                             'delta/(delta+1))')

        # Sets the parameter values from the given epsilon and delta
        cls.epsilon = epsilon
        cls.delta = delta

    @classmethod
    def get_w_mat(cls, sA, **kwargs):
        """Creates the W matrix from the adjacency matrix.

        The W matrix, when constructed from an adjacency matrix,
        is defined to be a matrix of the same dimensions where:
            - 0s are replaced by -1-delta
            - 1s are replaced by -1+epsilon
            - diagonals are kept as 0 regardless of the above

        Parameters
        ----------
        sA : array-like
            The adjacency matrix to create the W matrix from.
        **kwargs : dict
            Optionally, can include arguments for epsilon or delta
            if you don't want to change them globally and just want it
            for one run.

        Returns
        -------
        W : array-like
            The constructed W matrix.
        """

        # Check that the given adjacency matrix is valid.
        sA = cls._check_adjacency(sA)

        # Set epsilon and delta if given, otherwise use global values
        e = kwargs.get('epsilon') if kwargs.get('epsilon') else cls.epsilon
        d = kwargs.get('delta') if kwargs.get('delta') else cls.delta

        # Create the W matrix using the established shortcut formula.
        W = sA * (-1 + e) + (1 - sA) * (-1 - d)

        # Replace the diagonals of the constructed W with zeroes to
        # finalize its construction.
        np.fill_diagonal(W, 0)

        # Return the constructed W matrix.
        return W

    @classmethod
    def check_fp(cls, sA, sig: list, **kwargs):
        """Checks if a given subgraph (sigma) is a fixed point support
        of a given CTLN

        Uses the computations from "Predicting neural network dynamics
        via graphical analysis" by Katherine Morrison and Carina Curto
        found in Section 2.

        This section states that a given sigma is a fixed point support
        in a CTLN iff:
            x_sig = ((I-W_sig)^{-1})(b_sig)

        This method uses this principle to check a given sigma for a CTLN.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        sig : array-like
            The sigma/subgraph of the CTLN to check.
        b : array-like, optional
            The b vector to use. (Defaults to a column of 1s)

        Returns
        -------
        is_fp : bool
            A boolean that states whether or not the sigma is a fixed
            point support of the given CTLN.
        x_sig : array-like
            A column vector representing the firing rate of each neuron
            in the system in the course of the fixed point.
        """

        # Validates the provided adjacency matrix and constructs the W
        # matrix for it
        sA = cls._check_adjacency(sA)
        W = cls.get_w_mat(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = W.shape[0]

        # Let is_fp be True (assume it is a fixed point to begin).
        is_fp = True

        # Allows the user to provide their own b and default to a column
        # of 1s otherwise.
        if kwargs.get('b') is not None:
            b = kwargs['b']
        else:
            b = np.ones((n, 1))

        # Computes the I-W portion of the computation. Stores the result
        # as the matrix M
        M = np.subtract(np.identity(n), W)

        # Restricts the matrix M to sigma as required by the computation
        # to get I-W_sig.
        M_sig = M[sig, :][:, sig]

        # Creates the fixed point vector x_sig by multiplying the
        # inverse of the I-W_sig (aka (I-W_sig)^{-1}) by the b vector
        # restricted to sigma (b_sig)
        x_sig = np.linalg.inv(M_sig) @ b[sig, :]

        # Creates an empty column vector to store the firing rates for
        # the fixed point once computed
        x_fp = np.zeros((n, 1))

        # Creates the firing rates vector using the computed x_sig
        x_fp[sig, :] = x_sig

        # Declares the sigma to be *not* a fixed point if any of the
        # neurons in sigma (x_fp[sig,:]) are not "on" or "active" (<=0)
        if any(x_fp[sig, :] <= 0):
            is_fp = False

        # Checks that neurons *outside* of the sigma are *not* "on" or
        # "active"
        else:
            # Let sigbar be the set of nodes *not* in sigma
            sigbar = np.setdiff1d(list(range(n)), sig)

            # check each node in sigbar (nodes *not* in sigma)
            for k in sigbar:

                # Calculate the firing rate of the node (which is *not*
                # in sigma)
                sk = (W[k, sig] @ x_sig) + b[k, :]

                # If any node outside of sigma has a positive firing
                # rate (and is thus "active" or "on"), declare sigma
                # *not* a fixed point support
                if sk[0] > 0:
                    is_fp = False
                    break

        # Returns whether or not sigma is a fixed point support and the
        # firing rates of all of the neurons in the fixed point.
        return [is_fp, x_fp]

    @classmethod
    def check_stability(cls, sA, sig):
        """Checks whether a given fixed point support (sigma) is a
        stable fixed point or not (unstable).

        A fixed point is stable precisely when all eigenvalues of
        -I+W_sigma have a negative real part, and are unstable otherwise.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        sig : array-like
            The sigma/subgraph of the CTLN for which we want to check
            the stability of its corresponding fixed point.

        Returns
        -------
        stable : bool
            A boolean that states whether or not the fixed point is
            stable. (True = Stable, False = Unstable)
        eigvals : array-like
            The set of eigenvalues of the matrix -I+W_sig
        """

        # Validates the given adjacency matrix and constructs the
        # corresponding W matrix
        sA = cls._check_adjacency(sA)
        W = cls.get_w_mat(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = W.shape[0]

        # Computes the -I + W_sig matrix and stores it in M
        M = -1 * np.identity(len(sig)) + W[sig, sig]

        # Computes the eigenvalues of the M matrix
        eigvals = np.linalg.eig(M).eigenvalues

        # Gets the largest real part of the eigenvalues and stores as
        # lambda_max
        lambda_max = max(np.real(eigvals))

        # If the largest real part of the eigenvalues is negative,
        # *all* of them have negative real part, and the support is
        # stable. Otherwise, the largest real part of the eigenvalues is
        # positive, and the support is unstable.
        if lambda_max < 0:
            stable = True
        else:
            stable = False

        # Returns the stability boolean for the sigma and the
        # eigenvalues of the M matrix
        return [stable, eigvals]

    @classmethod
    def get_fp(cls, sA):
        """A method that finds all of the fixed points, their supports,
        and their stability for a given CTLN.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        fixpts : array-like
            The set of all fixed points of the CTLN.
        stability : array-like
            The stability of each fixed point in the CTLN.
        supports : array-like
            The fixed point support for each fixed point in the CTLN.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in sA,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Create empty lists to store our results for fixpts, stability,
        # and supports
        fixpts = []
        stability = []
        supports = []

        # For each sized subgraph...
        # (collections of 1,2,...,n nodes respectively)
        for k in range(n):

            # Get all possible combinations of nodes
            # (all possible subgraphs with that many nodes)
            subgraphs = list(combinations(list(range(n)), k + 1))

            # For each possible subgraph of the CTLN...
            for i in range(len(subgraphs)):

                # Let sigma be the particular subgraph we want to check
                sig = subgraphs[i]

                # Check if sigma is a fixed point
                is_fp, x_fp = cls.check_fp(sA, sig)

                # If this sigma *is* a fixed point, add it to the fixpts
                # list, add its support to the supports list, and add
                # its stability to the stability list
                if is_fp:
                    fixpts.append(np.transpose(x_fp))
                    t_sig = np.array(sig) + 1
                    supports.append(t_sig.tolist())
                    stability.append(cls.check_stability(sA, sig)[0])

        # Return the list of fixpts, supports, and stability
        return [fixpts, supports, stability]

    @classmethod
    def threshlin_ode(cls, sA, t, x0, **kwargs):
        """A method for solving the system of piecewise linear ordinary
        differential equations to get the firing rates of the neurons
        over time for a given set of initial conditions

        Uses the defined system of ODEs where:
            (x_i)' = -x_i + [sum_{j=1}^n (W_{ij}x_j + theta)]_{+}
        for i = 1, ..., n

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        t : float
            The time period to calculate up to (ie. solve the ode for
            times in interval [0,t])
        x0 : array-like
            The vector of initial conditions (starting firing rates) for
            each neuron in the CTLN.
        b : array-like, optional
            Allows the user to set the b vector manually (Defaults to a
            column of 1s)

        Returns
        -------
        W : array-like
            The constructed W matrix
        b : array-like
            The b vector that was used
        t : float
            The t value that was used
        x0 : array-like
            The vector of initial conditions that was used
        soln_y : list
            A list of all of the y values the ODE solved for
        soln_time : list
            A list of all the time values that correspond to the
            computed values in soln_y
        sA : array-like
            The adjacency matrix that was used.
        """

        # Validate and convert the adjacency matrix given and construct
        # the corresponding W matrix
        sA = cls._check_adjacency(sA)
        W = cls.get_w_mat(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = W.shape[0]

        # Allows the user to provide their own b and default to a column
        # of 1s otherwise.
        if kwargs.get('b') is not None:
            b = kwargs['b']
        else:
            b = np.ones((n, 1))

        # Let m be the length of the b vector
        m = b.shape[1]

        # Create empty lists to store the computed (t,y) pairs in parallel
        soln_y = []
        soln_time = []

        # Set the initial time value to 0
        t0 = 0

        # Construct the threshold nonlinearity function that turns any
        # negative firing rates to zeros.
        def _nonlin(x):
            # Just replaces negative firing rates with zeroes, leaves
            # the rest untouched.
            fixed_x = x
            fixed_x[fixed_x < 0] = 0
            return fixed_x

        for i in range(m):
            # Builds the differential equations for solving
            def _model(t, x):
                return -x + _nonlin(np.dot(W,x) + b[i, :])

            # Defines the time interval to solve for
            tspan = np.arange(t0, t0 + t + 0.01, 0.01)

            # Solves the system of ODEs
            sol = solve_ivp(
                _model,
                (tspan[0], tspan[-1]),
                x0,
                t_eval=tspan,
                method='RK45'
            )
            soln_y = sol.y
            soln_time = sol.t

        # Returns the results of the computation
        return [W, b, t, x0, soln_y, soln_time, sA]

    @classmethod
    def get_soln(cls, sA, **kwargs):
        """A method for obtaining the solution for a CTLN

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        theta : int, optional
            The value to use for theta (Defaults to 1)
        t : int, optional
            The endpoint of the time interval to calculate (Defaults to
            100)
        x0 : array-like, optional
            The initial firing rates of each neuron in the CTLN. (
            Defaults to random values between 0 and 0.1)
        b : array-like, optional
            The b vector to use (Defaults to a column of 1s times theta)

        Returns
        -------
        soln : array-like
            A list of the returned values from cls.threshlin_ode()
        """

        # Validate and convert the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Allow theta to be user defined, otherwise default to 1
        if kwargs.get('theta') is not None:
            theta = kwargs['theta']
        else:
            theta = 1

        # Allow t to be user defined, otherwise default to 100
        if kwargs.get('t') is not None:
            t = kwargs['t']
        else:
            t = 100

        # Allow x0 to be user defined, otherwise default to random
        # values 0 to 0.1
        if kwargs.get('x0') is not None:
            x0 = kwargs['x0']
        else:
            x0 = (np.zeros((1,n)) + (0.01 * np.random.uniform(
                size=n))).flatten().tolist()

        # Allow b to be user defined, otherwise default to a column of
        # 1s multiplied by theta
        if kwargs.get('b') is not None:
            b = kwargs['b']
        else:
            b = theta * np.ones((n, 1))

        # Compute and return the solution to the system of ODEs
        return cls.threshlin_ode(sA=sA, b=b, t=t, x0=x0)

    @classmethod
    def plot_graph(cls, sA, ax=None, show=True):
        """A method that plots the graph of the CTLN.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        ax : matplotlib.axes.Axes, optional
            The axes to plot the graph on. (Defaults to creating a new one)
        show : bool, optional
            Whether to show the graph after creation. (Defaults to True)
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Creates an axis for plotting if none provided
        if ax is None:
            plt.figure()
            ax = plt.gca()

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Creates a set of colors for the nodes in the graph
        colors = cls._get_graph_colors(n)

        # Defines a radius and determines the position of each node in
        # the graph to distribute them uniformly around the center.
        r = 1
        idxs = np.array(list(range(n)))
        x = r * np.cos(-(idxs) * 2 * np.pi / n + np.pi / 2)
        y = r * np.sin(-(idxs) * 2 * np.pi / n + np.pi / 2)

        # Draws an arrow between every pair of nodes where one exists in
        # the adjacency matrix.
        for j in range(n):
            for i in range(n):
                if sA[i, j] > 0:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    ax.arrow(x[j], y[j], dx * 0.87, dy * 0.87,
                             width=0.01,
                             head_width=0.07, head_length=0.1,
                             ec="#000000", fc='#000000')

        # Draws the nodes themselves using the colors and positions
        # determined earlier
        ax.scatter(x, y, c=colors, s=100)

        # Frames the graph properly
        ax.set_xlim(-1.2 * r, 1.3 * r)
        ax.set_ylim(-1.2 * r, 1.3 * r)

        # Removes unnecessary tick marks
        ax.set_yticks([])
        ax.set_xticks([])

        # If desired, shows the graph after completing it.
        if show: plt.show()

    @classmethod
    def plot_soln(cls, sA):
        """A method that plots both the graph and the solution of the CTLN.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Gets the solution for the given CTLN
        soln = cls.get_soln(sA)

        # Creates the list of colors to use for plotting
        colors = cls._get_graph_colors(n)

        # Creates the figure and axes for plotting
        fig, axs = plt.subplot_mosaic(
            '''
            GG12
            GG34
            BBBB
            DDDD
            '''
        )
        fig.set_size_inches((9,6))
        ax1 = axs['G']
        ax2 = [axs['1'], axs['2']]
        ax3 = [axs['3'], axs['4']]
        ax4 = axs['B']
        ax5 = axs['D']

        # Plots the graph portion
        cls.plot_graph(sA, ax=ax1, show=False)

        # Plots the projections of the solution
        proj_dir = cls.get_projection_direction(sA)
        cls.plot_projection(sA, tstart=0,tstop=25, direction=proj_dir, ax=ax2[0], show=False)
        cls.plot_projection(sA, tstart=25,tstop=50, direction=proj_dir, ax=ax2[1], show=False)
        cls.plot_projection(sA, tstart=50,tstop=75, direction=proj_dir, ax=ax3[0], show=False)
        cls.plot_projection(sA, tstart=75,tstop=100, direction=proj_dir, ax=ax3[1], show=False)

        # Plot Grayscale
        cls.plot_grayscale(soln[4], ax=ax4)

        # Plots the solution graph and its legend
        patches = []
        for i in range(n):
            ax5.plot(soln[5], soln[4][i], color=colors[i])
            patches.append(
                mlines.Line2D([], [], color=colors[i], label=f'{i + 1}'))
        plt.legend(
            handles=patches,
            frameon=False,
            ncol=n,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.5),
            title='Neuron'
        )

        # Adds axis labels for the solution graph
        ax5.set_ylabel('Firing Rate')
        ax5.set_xlabel('Time')

        # Adjusts the spacing of the subplots to prevent overlap and improve aesthetics
        plt.subplots_adjust(
            left=0.12,
            bottom=0.162,
            right=0.946,
            top=0.965,
            wspace=0.292,
            hspace=0.477
            )

        # Displays the figure
        plt.show()

    # Alias for plot_soln that was used in prior code. Continued here
    # for ease of use.
    run_ctln_model_script = plot_soln

    @classmethod
    def is_uid(cls, sA):
        """A method for seeing if a CTLN is uniform in-degree

        Checks that all of the row sums are equal

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        bool
            True if the CTLN is uniform in-degree, False otherwise
        """
        sA = cls._check_adjacency(sA)
        return len(np.unique(np.sum(sA, axis=1))) == 1

    @classmethod
    def is_uod(cls, sA):
        """A method for seeing if a CTLN is uniform out-degree

        Checks that all of the column sums are equal

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        bool
            True if the CTLN is uniform out-degree, False otherwise
        """
        sA = cls._check_adjacency(sA)
        return len(np.unique(np.sum(sA, axis=0))) == 1

    @classmethod
    def is_core(cls, sA):
        """A method for seeing if a CTLN is core.

        A core motif is a CTLN that has exactly one fixed point support
        which includes every node of the graph.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        is_core : bool
            True if the CTLN is a core motif, False otherwise
        """

        # Validate and convert the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Get the list of fp supports for the CTLN
        supports = cls.get_fp(sA)[1]

        # Default to assuming the CTLN is *not* core
        is_core = False

        # If there is only one support with all of the nodes, change
        # is_core to True
        if len(supports) == 1 and len(supports[0]) == n:
            is_core = True

        # Return the boolean of whether or not the CTLN is a core motif.
        return is_core

    @classmethod
    def is_permitted(cls, sA):
        """A method for seeing if a CTLN is permitted.

        A permitted motif is a CTLN that has a fixed point support
        containing every node of the graph (though this support is not
        necessarily the *only* one, as in core motifs).

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        is_permitted : bool
            True if the CTLN is a permitted motif, False otherwise
        """

        # Validate and convert the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Get the list of fp supports for the CTLN
        supports = cls.get_fp(sA)[1]

        # Checks if any fixed point support contains all nodes
        is_permitted = np.any([len(sup) == n for sup in supports])

        # Return the boolean of whether or not the CTLN is a
        # permitted motif.
        return is_permitted

    @classmethod
    def find_graphical_domination(
            cls,
            sA,
            types_to_look_for=(
                'inside-in',
                'outside-in',
                'inside-out',
                'outside-out'
            )
    ):
        '''A method for finding domination relationships within a CTLN.

        This is specific to *graphical* domination. This is defined such
        that:

        a node k "graphically dominates" a node j with respect to a
        subgraph sigma in G if
        1. For all i in sigma excluding j and k, if i -> j then i -> k
        2. If j in sigma, then j -> k
        3. If k in sigma, then k -/> j

        For determining the type of domination, the following rules hold:
        if j and k are in sigma: inside-in
        if j is in sigma but k is not: outside-in
        if k is in sigma but j is not: inside-out
        if j and k are not in sigma: outside-out

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        types_to_look_for : array-list, optional
            A list of the "types" of domination to look for. (Defaults
            to all possible types, but exists as an option in case user
            wants to ignore certain types of domination)

        Returns
        -------
        all_k : array-like
            A list of all the dominator nodes
        all_j
            A list of all the dominated nodes
        all_sigma
            A list of the sigmas that each k dominates j with respect to
        all_dom_type
            A list of the type of domination found for each domination
            relationship we have
        '''

        # Validate and convert the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Create empty lists to store results in
        all_k=[]
        all_j=[]
        all_sigma=[]
        all_dom_type=[]

        # For each sized subgraph...
        # (collections of 1,2,...,n nodes respectively)
        for i in range(n):
            # Get all possible combinations of nodes
            # (all possible subgraphs with that many nodes)
            subgraphs = list(combinations(list(range(n)), i + 1))

            # For each sub graph...
            for l in range(len(subgraphs)):
                sigma = np.array(subgraphs[l])

                # Get all pairs of nodes to check if one dominates the
                # other
                j_k_pairs = list(permutations(list(range(n)), 2))

                # for each j-k pair
                for j,k in j_k_pairs:

                    # Get all of the nodes in sigma other than j or k to
                    # check the first condition
                    sig_no_j_k = np.setdiff1d(sigma,[j,k]).flatten()

                    # If j and k are the same node, just move on (cannot
                    # dominate yourself)
                    if j == k:
                        continue

                    # If any of those i nodes send to j but not to k,
                    # no domination, move on.
                    # (fails 1st condition)
                    if np.any([sA[k,eye]==0 for eye in sig_no_j_k
                                if sA[j,eye]==1]):
                        continue

                    # If j is in sigma but does not send to k, move on
                    # (fails 2nd condition)
                    if j in sigma and sA[k,j]==0:
                        continue

                    # If k is in sigma and *does* send to j, move on
                    # (fails 3rd condition)
                    if k in sigma and sA[j,k]==1:
                        continue # if k in sigma, k must NOT send to j

                    # If we reach this point, we know that k dominates j
                    # with respect to sigma

                    # Sets the domination type using the rules by
                    # whether j or k are in sigma
                    dom_type = None
                    if j in sigma and k in sigma: dom_type = \
                        'inside-in'
                    elif j in sigma and k not in sigma: dom_type = \
                        'outside-in'
                    elif k in sigma and j not in sigma: dom_type = \
                        'inside-out'
                    elif j not in sigma and k not in sigma: dom_type = \
                        'outside-out'

                    # If the domination type we found is one we want to
                    # store (per user input), add the results to the
                    # lists we made earlier
                    if dom_type in types_to_look_for:
                        all_k.append(k+1)
                        all_j.append(j+1)
                        all_sigma.append(sigma+1)
                        all_dom_type.append(dom_type)

        # Return the completed lists
        return [all_k, all_j, all_sigma, all_dom_type]

    @classmethod
    def is_strongly_connected(cls,sA):
        '''A method for determining if a CTLN is strongly connected.

        The easiest way to find if a digraph is strongly connected is to
        use its reachability matrix. This is computed as (sA + I)^n

        If this matrix has all positive entries (including the
        diagonal), then the digraph is strongly connected.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        True if the CTLN is strongly connected, False otherwise.
        '''

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Calculates the reachability matrix (sA-I)^n
        M = np.linalg.matrix_power((sA + np.eye(n)),n)

        # Returns true if the reachability matrix is all positive.
        return np.all(M > 0)

    @classmethod
    def is_weakly_connected(cls, sA):
        '''A method for determining if a CTLN is weakly connected.

        The easiest way to find if a digraph is weakly connected is to
        find the underlying non-directed graph and check if *it* is
        connected.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        True if the CTLN is weakly connected, False otherwise.
        '''

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Makes sA symmetric (as if it were an undirected graph)
        sA = sA + np.transpose(sA)

        # Replaces any non-zeroes we created with 1s to get back to a
        # valid adjacency matrix
        sA[sA != 0] = 1

        # Uses the previous method to see if this underlying graph is
        # connected
        return cls.is_strongly_connected(sA)

    @classmethod
    def is_connected(cls, sA):
        """A method for determining if a CTLN is connected.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        A list of the form (is_weakly_connected, is_strongly_connected)
        """
        sA = cls._check_adjacency(sA)
        return [cls.is_weakly_connected(sA), cls.is_strongly_connected(sA)]

    @classmethod
    def is_strongly_core(cls, sA):
        """A method for determining if a CTLN is strongly core.

        A CTLN is strongly core if it is both a core motif and each of
        it's subgraphs that are not fixed point supports (all but the
        largest) are ruled out by some form of graphical domination.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        True if the CTLN is strongly core, False otherwise.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # If it is not core, we can safely say it is not strongly core
        if not CTLN.is_core(sA):
            return False

        # Get any domination relationships found
        all_k, all_j, all_sigma, all_dom_type = (
            cls.find_graphical_domination(sA)
        )

        # Filter down to those that are outside-in or inside-in since
        # these are the ones that 'kill' fixed points
        all_sigma = [tuple(a) for i,a in enumerate(all_sigma) if
                     all_dom_type[
            i] in ['outside-in','inside-in']]

        # Get a list of all possible sigmas
        pos_sigmas = []
        for i in range(n):
            pos_sigmas = pos_sigmas + list(combinations(
                list(range(n)), i + 1))

        # Make sure all but one (the maximal sigma) have that domination
        # 'killing' the fixed point.
        return (len(np.unique(np.asarray(all_sigma, dtype=object))) ==
                len(pos_sigmas)-1)

    @classmethod
    def is_hamiltonian(cls, sA):
        """A method for determining if a CTLN is hamiltonian
        (Contains a hamiltonian cycle of size n).

        To do this, we get every possible hamiltonian cycle of size n
        and check if it is found in the adjacency matrix.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        True if the CTLN is hamiltonian, False otherwise.

        ham_cycle : array-like
            A list of the hamiltonian cycles that were found in the graph.
        """

        # Validates and converts the adjacency matrix given
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Get a list of the possible size n hamiltonian paths, adding the
        # starting node to the end to make the paths into cycles
        orders = list(permutations(list(range(n))))
        for i,tup in enumerate(orders):
            orders[i] = tup + (tup[0],)

        # Assume each hamiltonian cycle is present in the given CTLN
        is_a_ham_cycle = [True]*len(orders)

        # For each possible hamiltonian cycle (to check)
        for i,cycle_to_check in enumerate(orders):
            for j in range(n):

                # Check each node in the cycle sends to the next
                n1 = cycle_to_check[j]
                n2 = cycle_to_check[j+1]
                if not sA[n2,n1] == 1:

                    # if it does not, mark that possible cycle as not
                    # appearing in the graph
                    is_a_ham_cycle[i] = False
                    break

        # Grabs the cycles that were found and removes the extra index
        # we added, as well as adding 1 to each index to make it human
        # readable
        ham_cycles = [
            [a+1 for a in order][:-1]
            for i,order
            in enumerate(orders)
            if is_a_ham_cycle[i]
        ]

        # return true if any possible hamiltonian cycle was found in the
        # given graph and return the list of hamiltonian cycles we found
        return sum(is_a_ham_cycle) > 0, ham_cycles

    @classmethod
    def get_projection_direction(cls, sA):
        """A method for generating a random projection direction.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        direction_vector : array-like
            A vector of size n by 2 that gives the two random directions to
            project onto.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Generates two vectors in random directions of n-dim space and normalizes them.
        d1 = np.random.uniform(0,1,size=(n,1))
        d1 = d1/sum(d1)
        d2 = np.random.uniform(0,1,size=(n,1))
        d2 = d2/sum(d2)

        # Concatenates the two direction vectors into one
        direction_vector = np.concatenate((d1,d2), axis=1)

        # Returns the direction vector for the projection
        return direction_vector

    @classmethod
    def plot_projection(cls, sA, tstart=40, tstop=80, direction=None, ax=None, show=True):
        """A method for plotting the projection of the solution onto two
        dimensions.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        tstart : int, optional
            The starting time index for the projection. (Defaults to 40)
        tstop : int, optional
            The stopping time index for the projection. (Defaults to 80)
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. (Defaults to creating a new one)
        show : bool, optional
            Whether to show the graph after creation. (Defaults to True)
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Gets the solution for the given CTLN
        soln = cls.get_soln(sA)

        # Creates the figure and axes for plotting if none provided
        if ax is None:
            _, ax = plt.subplots(figsize=(6,6))

        # Gets the projection direction to use for the projection.
        if direction is None:
            direction_vector = cls.get_projection_direction(sA)
        else:
            direction_vector = direction

        # Projects the solution onto the two random direction vectors.
        projection = soln[4].T @ direction_vector

        # Plots the projection
        ax.plot(projection[tstart*100:tstop*100,0], projection[tstart*100:tstop*100,1], color="black")

        # Gets rid of useless tick labels.
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Show the graph if desired.
        if show: plt.show()

    @classmethod
    def plot_grayscale(cls, soln, ax):
        """A method for plotting the solution in grayscale.

        Parameters
        ----------
        soln : array-like
            The solution to plot, as returned by cls.get_soln()
        ax : matplotlib.axes.Axes
            The axes to plot on.
        """

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = soln.shape[0]

        # Set color limits
        clim = [0,1]

        # Plot grayscale chart
        im = ax.imshow(soln, aspect='auto', cmap='gray_r', vmin=clim[0], vmax=clim[1],
               extent=[0, soln.shape[1], n, 0], origin='upper', interpolation='nearest')
        
        # Fix ticks and labels
        ax.set_xticks([])
        ax.set_ylabel('Neuron Number')
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(np.arange(1,n+1))
        ax.set_xlabel('Time')

    @classmethod
    def parallel_run(cls, matrices, method, num_processes=None):
        """Apply a CTLN class method to a list of matrices in parallel.
        
        This method uses multiprocessing to parallelize operations on multiple
        adjacency matrices, allowing for efficient batch processing without
        requiring users to set up parallelization themselves.
        
        Parameters
        ----------
        matrices : list
            A list of adjacency matrices (array-like) to process.
        method : callable
            A CTLN class method (e.g., CTLN.is_core, CTLN.get_fp,
            CTLN.is_strongly_connected).
        num_processes : int, optional
            Number of processes to use. If None (default), uses all available
            CPU cores.
        
        Returns
        -------
        results : list
            A list of results from applying the method to each matrix, in the
            same order as the input matrices.
        """
        
        # Determine number of processes to use
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Use a process pool to parallelize the computation
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(method, matrices)
        
        # Return the results in the same order as the input matrices
        return results

    @staticmethod
    def _check_for_updates():
        """
        A method for checking if there is an update available for the py_ctln package.
        """

        # Define the package name to check
        package_name = 'py_ctln'

        try:
            # Get the currently used version of the package
            current_version = version(package_name)

            # Get the latest release version number from pip
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=1)
            latest_version = response.json()['info']['version']

            # Alert the user if there is an update available, otherwise do nothing
            if current_version != latest_version:
                print(f"Update available for {package_name}: {current_version} -> {latest_version}")
                print(f"Run 'pip install --upgrade {package_name}' to update.")

        except Exception as e:
            # If there was an error (e.g. no internet connection), just pass and do not alert the user.
            pass

    @classmethod
    def is_circulant(cls, sA):
        """ A method for determining if a CTLN is circulant.

        A circulant graph is one where there is some ordering of the nodes such
        that the adjacency matrix is circulant. A circulant matrix is one
        where each column is a shifted version of the previous column.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.

        Returns
        -------
        is_circulant : bool
            True if the CTLN is circulant, False otherwise.
        ordering : array-like
            The ordering of the nodes that makes the adjacency matrix circulant
        forward_edges : array-like
            The pattern used for circulant graph notation
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Check every possible ordering of the nodes
        for perm in permutations(range(1,n)):

            # Get current ordering to check
            current_perm = [0] + list(perm)

            # Get the adjacency matrix in that ordering
            sA_new = sA[np.ix_(current_perm, current_perm)]

            # Shift each column as appropriate
            sA_shifted = np.zeros_like(sA_new)

            # Check if matrix is circulant
            for j in range(n):
                sA_shifted[:,j] = np.roll(sA_new[:,j],-j)

            # Get the first column
            first_col = sA_shifted[:,0]

            # Check that the proper pattern appears
            if np.allclose(sA_shifted, np.tile(first_col.reshape(-1,1), (1,n))):

                # Return results
                return True, [i+1 for i in current_perm], np.nonzero(first_col)[0].tolist()
        
        # If not circulant, return false
        return False, [], []
    
    @classmethod
    def is_clique_union(cls, sA, tau_partition):
        """ A method for determining if a CTLN is a clique union.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        tau_partition : list of lists
            A partition of the nodes into subsets.
        
        Returns
        -------
        True if the CTLN is a clique union with respect to the given partition, False otherwise.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Get number of parts in the partition
        n_parts = len(tau_partition)

        # Check each part
        for i in range(n_parts):
            for j in range(i+1, n_parts):

                # Check each pair of parts within the partition is bidirectionally connected
                tau_a = np.asarray(tau_partition[i])
                tau_b = np.asarray(tau_partition[j])
                expected = tau_a.size * tau_b.size

                # If any pair of parts of the partition are not, return False.
                if sA[np.ix_(tau_a, tau_b)].sum() != expected or sA[np.ix_(tau_b, tau_a)].sum() != expected:
                    return False
                
        # Return true if clique union.
        return True

    @classmethod
    def is_directional(cls, sA, omega, tau):
        """ A method for determining if a CTLN is directional with respect to a given 
            omega and tau partition.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        omega : array-like
            A subset of the nodes in the CTLN.
        tau : array-like
            A subset of the nodes in the CTLN, disjoint from omega and together with omega 
            containing all nodes in the CTLN.

        Returns
        -------
        directional : bool
            True if the CTLN is directional with respect to the given omega and tau, False otherwise
        sigma_fail : list
            If the CTLN is not directional, a list of the nodes in the sigma that fails the directional condition.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in W,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Turn omega and tau into sets
        omega_s = set(omega)
        tau_s = set(tau)

        # Ensure omega and tau are disjoint
        if len(omega_s.intersection(tau_s))>0:
            raise ValueError("Omega and Tau must be disjoint.")
        
        # Ensure omega and tau together contain all nodes
        if len(omega_s) + len(tau_s) < n:
            raise ValueError("Omega and Tau together must contain all nodes.")
        
        # Set default returns
        directional = True
        sigma_fail : list[int] = []

        # Check every sigma
        for mask in range(1, 1 << n):
            sigma = {i for i in range(n) if (mask >> i) & 1}

            # If sigma contains any node from omega
            if sigma & omega_s:

                # Check there is some form of domination, as required as by the definition, and return false if none found
                j_of_interest = sigma & omega_s
                all_k, all_j, all_sigma, _ = cls.find_graphical_domination(sA, types_to_look_for=['inside-in'])
                if not any(j-1 in j_of_interest and set(sigma) == set([i-1 for i in sigma_dom.tolist()]) for j, sigma_dom in zip(all_j, all_sigma)):
                    k_of_interest = set(range(n)) - sigma
                    all_k, all_j, all_sigma, _ = cls.find_graphical_domination(sA, types_to_look_for=['outside-in'])
                    if not any(k-1 in k_of_interest and j-1 in j_of_interest and set(sigma) == set([i-1 for i in sigma_dom.tolist()]) for k,j, sigma_dom in zip(all_k, all_j, all_sigma)):
                        directional = False
                        sigma_fail = sorted(sigma)
                        return directional, sigma_fail
        
        # If domination conditions hold, return true
        return directional, sigma_fail
    
    @staticmethod
    def get_circulant_graph(n, forward_edges):
        """ A method for generating the adjacency matrix of a circulant graph.

        Parameters
        ----------
        n : int
            The number of nodes in the circulant graph.
        forward_edges : list of ints
            A list of the indices of the forward edges in the circulant graph. 
            For example, if forward_edges = [1, 3], then each node i sends to nodes (i+1) mod n and (i+3) mod n.

        Returns
        -------
        sA : array-like
            The adjacency matrix of the circulant graph.
        """

        # Create an empty adjacency matrix
        sA = np.zeros((n,n), dtype=int)

        # Fill in the adjacency matrix according to the forward edges
        for i in range(n):
            for edge in forward_edges:
                sA[(i + edge) % n, i] = 1
        
        # Return the adjacency matrix
        return sA

    @classmethod
    def get_symbolic_w(cls, sA):
        # Check that the given adjacency matrix is valid.
        sA = cls._check_adjacency(sA)

        e, d = sp.symbols('e d')

        # Create the W matrix using the established shortcut formula.
        W = sA * (-1 + e) + (1 - sA) * (-1 - d)

        # Replace the diagonals of the constructed W with zeroes to
        # finalize its construction.
        np.fill_diagonal(W, 0)

        # Return the constructed W matrix.
        return W

    @classmethod
    def s_i_of_sig(cls,sA,sigma,i,**kwargs):
        """ A method for calculating the s_i value for a particular sA, i,
        and sigma

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        sigma : list of ints
            A list of the indices for the nodes in the sigma to
            calculate with respect to
        i : int
            The index of the node to check as the i
        kwargs : dict
            Additional keyword arguments to pass to get_w_mat, 
            for values of epsilon and delta.
        
        Returns
        -------
        The computed s_i^sigma value.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Gets the corresponding W matrix for sA
        W = cls.get_w_mat(sA, **kwargs)

        # Gets the set sigma union the node i
        sig_u_i = list(set(sigma).union({i}))

        # Gets the necessary identity matrix
        I = np.identity(len(sig_u_i))

        # Gets W restricted to sig_u_i
        W_sig_u_i = W[np.ix_(sig_u_i,sig_u_i)]

        # Calculates the I-W resticted to sig_u_i and then replaces the
        # i'th column with 1s
        M = I - W_sig_u_i
        ind = sig_u_i.index(i)
        M[:,ind] = 1

        # Returns the determinant of that matrix
        return np.linalg.det(M)

    @classmethod
    def check_si_survival(cls, sA, sigma, **kwargs):
        """ A method for determining if a sigma is a FP of a given CTLN
            using sign conditions.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN
        sigma : array-like
            A list of integer indices that represent which nodes are in sigma
        kwargs : dict
            Additional keyword arguments to pass to s_i_of_sig, 
            for values of epsilon and delta.

        Returns
        -------
        True if sigma survives, False if not.
        """

        # Validates and converts the given adjacency matrix
        sA = cls._check_adjacency(sA)

        # Let n be the size of the ctln (number of rows/columns in sA,
        # number of neurons, etc.)
        n = sA.shape[0]

        # Get list of si values for the sigma
        si_list = [cls.s_i_of_sig(sA, sigma, i, **kwargs) for i in list(range(n))]

        # Separate values into those in and outside of sigma
        si_list_inside = [i for e,i in enumerate(si_list) if e in sigma]
        si_list_outside = [i for e,i in enumerate(si_list) if e not in sigma]

        # Get the sign for inside of sigma
        temp = np.sign(si_list_inside[0])

        # Make sure all inside match and all outside are opposite
        check_inside = np.all([np.sign(i)==temp for i in si_list_inside])
        check_outside = np.all([np.sign(i)!=temp for i in si_list_outside])

        # Return result
        return check_inside and check_outside
    
    @classmethod
    def get_fp_by_si(cls, sA, **kwargs):
        sA = cls._check_adjacency(sA)
        n = sA.shape[0]
        sigmas = list(chain.from_iterable(combinations(range(n), r) for r in range(len(range(n)) + 1)))
        del sigmas[0]
        res = [cls.check_si_survival(sA, s, **kwargs) for s in sigmas]
        return [np.asarray(s)+1 for s,t in zip(sigmas,res) if t]
    
    @classmethod
    def get_param_indep_fps(cls, sA):
        sA = cls._check_adjacency(sA)
        n = sA.shape[0]
        sigmas = list(chain.from_iterable(combinations(range(n), r) for r in range(len(range(n)) + 1)))
        del sigmas[0]



    @staticmethod
    def _make_all_embeddings(n):
        """ A method for producing all of the possible embeddings to test when checking parameter dependence

        Parameters
        ----------
        n : int
            The number of nodes in the graph

        Returns
        -------
        embeddings : array-like
            An array of size (2^n - 1) by n, where each row is a different embedding of the n nodes.
        """

        # Create an empty array to hold the embeddings
        embeddings = np.zeros((2**n, n), dtype=int)

        # Fill the array with the possible embeddings
        for i in range(2**n):
            embeddings[i] = [(i >> j) & 1 for j in range(n)]

        # Return all but the trivial embedding
        return embeddings[1:]
    
    @classmethod
    def _check_survival_param_ind(cls, sA, sigma,k_of_interest, dom):
        """ A method for checking if a particular sigma survives the addition of particular dominator nodes k.

        Parameters
        ----------
        sA : array-like
            The adjacency matrix of the CTLN.
        sigma : array-like
            The subgraph we are checking the survival of.
        k_of_interest : array-like
            The dominator nodes we are checking the survival with respect to.
        dom : dict
            A dictionary containing the domination relationships found in the graph, as returned by find_graphical_domination

        Returns
        -------
        tf_param_ind : array-like
            A boolean array of size len(k_of_interest) by 1, where each entry is True if the survival of sigma with respect to that k is parameter independent, and False otherwise.
        tf_survives : array-like
            A boolean array of size len(k_of_interest) by 1, where each entry is True if sigma survives the addition of that k with the current parameters, and False otherwise.
        tf_uid : bool
            A boolean indicating whether sigma is a uniform in-degree subgraph, which guarantees parameter independent survival.
        """

        # Validates and converts the given adjacency matrix
        if isinstance(sA, list):
            sA = cls._check_adjacency(sA)

        # Get the subgraph of sA corresponding to sigma
        sA_sig = sA[np.ix_(sigma, sigma)]

        # Create empty arrays to hold our results
        tf_param_ind = np.zeros((len(k_of_interest),1), dtype=bool)
        tf_survives = np.zeros((len(k_of_interest),1), dtype=bool)

        # Get the domination relationships needed for later
        all_k, _, all_sigma, dom_type = dom
        dom = zip(all_k, all_sigma, dom_type)
        
        # Make sure that sigma is a permitted motif to begin with, otherwise we cannot check its survival
        assert cls.is_permitted(sA_sig), "Sigma must be a permitted motif to begin with"
        
        # A mini UID check to see if sigma is UID
        def _sig_uid(sA, sigma):
            sA_sig = sA[np.ix_(sigma, sigma)]
            row_sums = sA_sig.sum(axis=1)
            tf = np.all(row_sums == row_sums[0])
            if tf:
                return True, row_sums[0]
            else:
                return False, 0

        # Check if sigma is UID
        tf_uid, d_in = _sig_uid(sA, sigma)

        # If sigma is UID...
        if tf_uid:
            # It will be parameter independent
            tf_param_ind = np.ones((len(k_of_interest),1), dtype=bool)

            # Check if it is target free to determine if it survives
            d_k = sA[np.ix_(k_of_interest,sigma)] @ np.ones((len(sigma),1))
            tf_survives = d_k<=d_in

            # Return the results
            return tf_param_ind, tf_survives, tf_uid
        
        # Otherwise if NOT UID... (Check domination)
        else: 
            # Check each possible dominator...
            for l,k in enumerate(k_of_interest):

                # See if sigma survives with current parameters
                sig_cup_k = np.append(sigma, k)
                tf_fp = cls.check_fp(sA[np.ix_(sig_cup_k, sig_cup_k)],list(range(len(sigma))))[0]

                # If it does survive...
                if tf_fp:
                    # Could domination kill it with different parameters
                    tf_survives[l] = True
                    if any(k-1 in k_of_interest and set(sigma) == set([i-1 for i in sigma_dom.tolist()]) for k, sigma_dom, dom_type in dom if dom_type == 'inside-out'):
                        tf_param_ind[l] = True
                
                # If it does not survive...
                else:
                    # Could domination save it with different parameters
                    tf_survives[l] = False
                    if any(k-1 in k_of_interest and set(sigma) == set([i-1 for i in sigma_dom.tolist()]) for k, sigma_dom, dom_type in dom if dom_type == 'outside-in'):
                        tf_param_ind[l] = True

        # Return results
        return tf_param_ind, tf_survives, tf_uid
    
    @classmethod
    def _check_survival_all_embeddings(cls, sA_sig, sigma):
        """ A method for checking the survival of a particular sigma with respect to all possible domination embeddings.

        Parameters
        ----------
        sA_sig : array-like
            The adjacency matrix of the subgraph corresponding to sigma.
        sigma : array-like
            The subgraph we are checking the survival of.

        Returns
        -------
        poss_param_dep_embeddings : array-like
            An array of embeddings that may lead to parameter dependence.
        """
        # Validates and converts the given adjacency matrix
        sA_sig = cls._check_adjacency(sA_sig)

        # Let n be the size of the matrix given
        n=sA_sig.shape[0]

        # Get all of the embeddings to check
        embeddings = cls._make_all_embeddings(n)

        # Count how many embeddings we need to check
        num_embed= embeddings.shape[0]

        # Create the larger adjacency matrix to check the embeddings in
        sA = np.zeros((n+num_embed,n+num_embed), dtype=int)

        # Fill in the top left corner with sA_sig
        sA[np.ix_(sigma, sigma)] = sA_sig

        # Get the possible dominator nodes
        k_of_interest = list(set(range(sA.shape[0])) - set(sigma))

        # Fill in the bottom left corner with the embeddings we are checking
        sA[np.ix_(k_of_interest, sigma)] = embeddings

        # Find domination relationships to pass to the survival check method
        dom = cls.find_graphical_domination(sA_sig, types_to_look_for=['inside-out', 'outside-in'])

        # Check the survival of sigma with respect to each possible dominator node, and whether that survival is parameter independent
        tf_param_ind, _, _ = cls._check_survival_param_ind(sA, sigma, k_of_interest, dom)

        # Select the embeddings that aren't definitely independent...
        idx_poss_param_dep = np.where(~tf_param_ind)[0]
        poss_param_dep_embeddings=sA[np.ix_(np.asarray(k_of_interest)[idx_poss_param_dep],sigma)]

        # And return them
        return poss_param_dep_embeddings
    
    @classmethod
    def core_is_param_indep(cls, sA):
        sA = cls._check_adjacency(sA)
        # We'll have to check each poss_param_dep_embedding symbollically to see if any create inequalities that could be satisfied by some parameters and not others.
        pass

    @classmethod
    def has_chaining_overlap(cls, sA,o1,t1,o2,t2):
        sA = cls._check_adjacency(sA)
        g1 = [o1,t1]
        g2 = [o2,t2]

        if set(t2)<=set(o2):
            if len(set(o1)&set(t2))==0:
                g2_minus_g1 = set(g2)-set(g1)
                o1_in_g2 = set(o1) & set([o2,t2])
                if sum(sum(sA[np.ix_(list(o1_in_g2),list(g2_minus_g1))])) == 0:
                    return True
        else:
            return False
        
    @classmethod
    def is_simply_added(cls, sA, tau, omega):
        sA = cls._check_adjacency(sA)
        sub_mat = sA[np.ix_(tau, omega)]
        col_sums = np.sum(sub_mat, axis=0)
        num_proj = len([c for c in col_sums if c == len(tau)])
        num_non_proj = len([c for c in col_sums if c == 0])
        if(num_non_proj+num_proj) == len(omega):
            return True
        else:
            return False
        
    @classmethod
    def find_simply_embedded_taus(cls, sA):
        sA = cls._check_adjacency(sA)
        n = sA.shape[0]
        se_taus = []

        for i in range(1,n):
            combos = combinations(range(n), i)
            for j,tau in enumerate(combos):
                omega = list(set(range(n)) - set(tau))
                if cls.is_simply_added(sA, tau, omega):
                    se_taus.append(tau)

        return se_taus

# Checks for package update on import
CTLN._check_for_updates()

sA = np.array([
    [0,0,0,1,1],
    [1,0,1,0,0],
    [1,1,0,0,0],
    [0,1,1,0,0],
    [0,1,0,1,0]
])
#print(CTLN.get_fp(sA))
#print(CTLN.check_si_survival(sA, [0,1,2,3,4]))
#print(CTLN.get_fp_by_si(sA))
print(CTLN.get_symbolic_w(sA))

#CTLN.set_params(epsilon=0.1, delta=0.12)
#CTLN.set_params(epsilon=0.25, delta=0.5)
#print(CTLN.find_graphical_domination(sA, types_to_look_for=['inside-out', 'outside-in']))
#print(CTLN._check_survival_all_embeddings(sA[np.ix_([0,1,2,3],[0,1,2,3])], [0,1,2,3]))
#print(CTLN._check_survival_param_ind(sA, [0,1,2,3], [4]))
#print(CTLN.get_fp(sA))