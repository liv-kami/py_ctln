# ─────────────────────────── Libraries ────────────────────────────

import numpy as np
from itertools import combinations, permutations
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pickle
from importlib.resources import files
from pathlib import Path


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
    """

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
        path_ref: Path = files("py_ctln.known_network_data") / (
            f"all_{n}.pkl")
        if not path_ref.exists():
            raise ValueError(f'Sorry, we do not yet have the list you '
                             f'requested: all_n({n})')
        return cls._load_data(path_ref)

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
        path_ref: Path = files("py_ctln.known_network_data") / (
            f"core_{n}.pkl")
        if not path_ref.exists():
            raise ValueError(f'Sorry, we do not yet have the list you '
                             f'requested: core_n({n})')
        return cls._load_data(path_ref)

    # TODO: Add more types/classes of CTLNs we can have lists of!

# ──────────────────────── Main Ctln Funcs ─────────────────────────

class CTLN:
    """A class used to provide functions for Combinatorial Threshold
    Linear Network (CTLN) calculations and research.

    ...

    Attributes
    ----------
    epsilon : float, optional
        The value to use for the epsilon parameter (default is 0.25).
    delta : float, optional
        The value to use for the delta parameter (default is 0.5).
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
    """

    #TODO: add new methods to docs here

    epsilon: float = 0.25
    delta: float = 0.5

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

        # Creates n colors evenlt distributed along the rainbow
        colors = plt.get_cmap('rainbow', n)
        colors = colors(np.linspace(0, 1, n))

        # Converts the colors to hex codes
        colors = [mcolors.to_hex(color) for color in colors]

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
    def set_params(cls, epsilon: float = 0.25, delta: float = 0.5):
        """Allows the user to define the values for the parameters
        epsilon and delta

        Parameters
        ----------
        epsilon : float, optional
            The value to use for the epsilon parameter (default is 0.25).
        delta : float, optional
            The value to use for the delta parameter (default is 0.5).
        """

        # Sets the parameter values from the given epsilon and delta
        cls.epsilon = epsilon
        cls.delta = delta

    @classmethod
    def get_w_mat(cls, sA):
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

        Returns
        -------
        W : array-like
            The constructed W matrix.
        """

        # Check that the given adjacency matrix is valid.
        sA = cls._check_adjacency(sA)

        # Create the W matrix using the established shortcut formula.
        W = sA * (-1 + cls.epsilon) + (1 - sA) * (-1 - cls.delta)

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
        m = len(b)

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
            to_fix = np.where(x < 0)
            fixed_x = x
            fixed_x[to_fix] = 0
            return fixed_x

        for i in range(m):
            # Builds the differential equations for solving
            def _model(t, x):
                return -x + _nonlin(W @ x + b[i, :])

            # Defines the time interval to solve for
            tspan = np.arange(t0, t0 + t + 0.01, 0.01)

            # Solves the system of ODEs
            sol = solve_ivp(
                _model,
                (tspan[0], tspan[-1]),
                x0,
                t_eval=tspan
            )

            # Builds the necessary results from the solution above
            time = sol.t
            x = sol.y.T
            with np.errstate(
                    divide='ignore',
                    invalid='ignore',
                    over='ignore'
            ):
                y = W @ np.transpose(x) + (
                    (b @ np.ones((1, len(np.transpose(x)[1]))))
                )
            soln_y = y
            soln_time = time

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
            # TODO: check the x0 generation - it is the uniform random
            #  stuff but it seems to always be really high which seems
            #  sketch to me
            #  could be related to the y axis scaling but I
            #  feel like it should not *always* start higher than it
            #  ends up...

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
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 8),
            height_ratios=[3, 1]
        )

        # Plots the graph portion
        cls.plot_graph(sA, ax=axs[0], show=False)

        # Plots the solution graph and its legend
        ax = axs[1]
        patches = []
        for i in range(n):
            ax.plot(soln[5], soln[4][i], color=colors[i])
            patches.append(
                mpatches.Patch(color=colors[i], label=f'{i + 1}'))
        plt.legend(
            handles=patches,
            frameon=False,
            ncol=n,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.05)
        )

        # Adds axis labels for the solution graph
        ax.set_ylabel('Firing Rate')
        ax.set_xlabel('Time')

        # Displays the figure
        plt.show()
        # TODO: look into adding the other windows for this script
        #   (projection and bars thing)

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
    def find_domination(
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

        # If there are domination relationships for each sigma other
        # than the maximal one, return True. Otherwise return False
        if len(
                np.unique([a for b in cls.find_domination(sA)[2] for a
                          in b])
        ) != len(
            [x for y in [list(combinations(list(range(n)),i+1)) for i in
                         range(n)] for x in y])-1:
            return True
        else:
            return False

        #TODO: Needs more proper testing :(

# ──────────────────────── To-do By v0.1.1 ─────────────────────────

# TODO: Gray scale and phase space projection stuff for
#  run_ctln_model_script (Actually going to move this to v0.1.1 since
#  there is a high probability of me screwing this up and wanting to
#  roll back lol)

# ─────────────────────── Caitlyn's Wishlist ───────────────────────

# TODO: is_cyc_union
# TODO: is_clique_union
# TODO: is_connected_union
# TODO: is_composite_graph (?)
# TODO: reduce graph (using domination)
# TODO: identify directional cycle covers and weakly directional covers
# TODO: identify simply-embedded partitions
# TODO: identify strong simply-embedded partitions
# TODO: identify/construct circulant graphs (with their notation I assume)
# TODO: find hamiltonian cycles/is_hamiltonian
# TODO: all_cycles function (not on her list but I'm adding it)
# TODO: identify firing sequence from ode solution
# TODO: construct graphs (cycles, composite, circulant, etc.)

# TODO: Look for more functionality to add! (After the rest lol)
#       (Can also check the ctln github to see if theres stuff there)

# TODO: Random size n graph generator
#  (also not from caitlyn, but I want it)

# TODO: Expand the documentation/wiki on github
