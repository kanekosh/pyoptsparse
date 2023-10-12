import numpy as np
from scipy.optimize import line_search as scipy_line_search
import copy

import cvxpy

from scipy.optimize._linesearch import LineSearchWarning
import warnings
warnings.simplefilter('ignore', LineSearchWarning)

"""
SQP algorithm

Author: Shugo Kaneko

Additional dependencies: scipy, cvxpy, gurobipy.
I use the following versions, but other versions probably works as well.
- scipy==1.10.0
- cvxpy-base==1.1.7
- gurobipy==10.0.2

TODO: dummy constraints are not yet implemented in elastic mode. Therefore, the following problems fail if it enters elastic mode:
- unconstrained
- bound-only
- inequality-only
- equality-only
"""

def _convert_array_largeval_to_None(array):
    """
    Convert very small or large float in the array to None
    Used to convert pyOptSparse style bound arrays into the style adopted in this code
    """
    threshold = 1e10
    array = list(array)
    for i in range(len(array)):
        if array[i] is not None:
            if array[i] < -threshold or array[i] > threshold:
                array[i] = None
    return array


class SQP():
    """
    Solve the following NLP problem

        min:        f(x)
        w.r.t.:     x
        subject to: cons_lb <= c(x) <= cons_ub
                    x_lb    <= x    <= x_ub

        where c(x) = [c_nonl(x); c_lin(x)] is the concatenation of nonlinear and linear constraints.

    The NLP is then converted to the following form:
        min:        f(x)
        w.r.t.:     x
        subject to: h(x) =  0
                    g(x) <= 0

                    where g = [(cons_lb - c), (c - cons_ub), (x_lb - x), (x - x_ub), dummy]
                    and   g_nonl := [(cons_lb - c), (c - cons_ub)] (nonlinear part)
                          g_lin  := [(x_lb - x), (x - x_ub)] (linear part)

    Elastic problem is:
        min:        f(x) + gamma * (sum(v + w) + sum(z))
        w.r.t.:     x, v, w, z
                    (new variable is defined as xvwz = [x, v, w, z])
        subject to: h - v + w = 0
                    g_nonl - z <= 0
                    x_lb <= x <= x_ub
                    v, w, z >= 0
    The constraints are reformatted as:
                    h_elastic = h - v + w = 0
                    g_elastic = [g_nonl - z, g_lin, -v, -w, -z] <= 0

    Exit status:
         -1: not started
         0:  Optimal solution found
        10:  Iteration limit exceeded
        20:  Hessian approx is not positive semi-definite
        30:  Line search failed
        40:  QP solver failed
        50:  Elastic mode failed to find a feasible point

    """

    def __init__(self, nx, nc, x_lb, x_ub, obj, grad_obj, cons=None, jac_cons=None, cons_lb=None, cons_ub=None, n_nonl_cons=None):
        """
        Sequential Quadratic Programming (SQP) algorithm.

        Parameters
        ----------
        nx : int
            Number of optimization variables
        nc : int
            Number of constraints
        x_lb : ndarray, shape (nx,)
            Lower bounds of variabes. An entry should be `None` or very small value (< -1e10) if unbounded.
        x_ub : ndarray, shape (nx,)
            Upper bounds of variabes. An entry should be `None` or very large value (> +1e10) if unbounded.
        obj : callable, f(x)
            Objective function
        grad_obj : callable, df/dx(x)
            Gradient of objective function
        cons : callable, c(x)
            Constraint functions (both qquality and inequality combined)
            Output of the callable should be a 1D array of shape (nc,), not a scalar even if nc=1.
        jac_cons : callable, dc/dx(x)
            Jacobian of constraints
            dc/dx should be a 2D array of shape (nh, nx), not 1D even if nh=1.
        cons_lb : ndarray, shape (nc,)
            Lower bounds of constraints. An entry should be `None` or very small value (< -1e10) if unbounded.
            For equality constraints, set cons_lb == cons_ub.
        cons_ub : ndarray, shape (nc,)
            Upper bounds of constraints. An entry should be `None` or very large value (> +1e10) if unbounded.
        n_nonl_cons : int
            Number of nonlinear constraints. If None, it is assumed that all constraints are nonlinear (i.e., later we set n_nonl_cons = nc).
            Otherwise, the constraint `cons` should be ordered by [c_nonl, c_lin].
            I.e., c[0:n_nonl_cons] are nonlinear, and c[n_nonl_cons:] are linear.

        TODO: automatically fix (reshape) constraints and Jacobian when nh=1 or ng=1.
        """

        # set default options
        self.set_options()

        # flag for elastic mode
        self.elastic_mode = False

        # setup objective function
        self._obj_func = obj
        self._obj_grad_func = grad_obj

        # convert pyOptSparse-style bound arrays (which includes large/small float for unbounded vars/cons) into the style adopted in this code
        x_lb = _convert_array_largeval_to_None(x_lb)
        x_ub = _convert_array_largeval_to_None(x_ub)
        if cons_lb is not None:
            cons_lb = _convert_array_largeval_to_None(cons_lb)
        if cons_ub is not None:
            cons_ub = _convert_array_largeval_to_None(cons_ub)

        # setup variable bounds as inequality constraints
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_lb_idx = []   # list of variable indices that has a lower bound. This will be imposed as x_lb - x <= 0
        self.x_lb_val = []
        self.x_ub_idx = []   # list of variable indices that has a upper bound. This will be imposed as x - x_ub <= 0
        self.x_ub_val = []
        for i in range(nx):
            if x_lb[i] is not None:
                self.x_lb_idx.append(i)
                self.x_lb_val.append(x_lb[i])
            if x_ub[i] is not None:
                self.x_ub_idx.append(i)
                self.x_ub_val.append(x_ub[i])
        # END FOR (adding variable bounds)
        # Jacobian of bound constraints
        self.jac_x_lb = -np.eye(nx)[self.x_lb_idx, :]
        self.jac_x_ub = np.eye(nx)[self.x_ub_idx, :]

        # setup constraints
        self.con_eq_idx = []   # constraint indices for equalities: c_val - c(x) = 0
        self.con_eq_val = []   # equality value
        self.con_ineq_lb_idx = []  # constraint indices for inequalities with lower bound: c_lb - c(x) <= 0
        self.con_ineq_lb_val = []  # lower bound value
        self.con_ineq_ub_idx = []  # constraint indices for inequalities with upper bound: c(x) - c_ub <= 0
        self.con_ineq_ub_val = []  # upper bound value

        # flag to classify problem characteristics
        self.flag_unconstrained = False    # uncon & unbounded
        self.flag_bounds_only = False      # only bound inequalities
        self.flag_inequality_only = False  # no equalities, with (non-bound) inequalities
        self.flag_equality_only = False    # no inequalities and bounds

        if cons is None and len(self.x_lb_idx) == 0 and len(self.x_ub_idx) == 0:
            # unconstrainted & unbounded problem.
            print('\n*** Unconstrained & unbounded problem. Setting up dummy constraints.')
            self.flag_unconstrained = True

            # setup dummy constraints (one equality and one inequality) that are always feasible
            self._cons_func = lambda x: np.array([0., -1.])
            self._jac_cons_func = lambda x: np.zeros((2, nx))
            self.con_eq_idx.append(0)
            self.con_eq_val.append(0.)
            self.con_ineq_ub_idx.append(1)
            self.con_ineq_ub_val.append(0.)

        elif cons is None:
            # only bound constraints
            print('\n*** Bound-only problem. Setting up a dummy equality constraint.')
            self.flag_bounds_only = True
            self._cons_func = lambda x: np.array([0.])
            self._jac_cons_func = lambda x: np.zeros((1, nx))
            self.con_eq_idx.append(0)
            self.con_eq_val.append(0.)

        else:
            self._cons_func = cons
            self._jac_cons_func = jac_cons

            # TODO (FFR): check/save constraint linearity here

            # loop over each constraint, and judge if it is equality or inequality
            for i in range(nc):
                if cons_lb[i] is None and cons_ub[i] is None:
                    pass
                
                else:
                    # add an equality constraint
                    if cons_lb[i] is not None and cons_ub[i] is not None:
                        if np.isclose(cons_lb[i], cons_ub[i], atol=1e-10):
                            self.con_eq_idx.append(i)
                            self.con_eq_val.append(cons_lb[i])
                            continue
                    
                    # otherwise, add an inequality constraint. Handle lower and upper bounds separately
                    if cons_lb[i] is not None:
                        self.con_ineq_lb_idx.append(i)
                        self.con_ineq_lb_val.append(cons_lb[i])
                    if cons_ub[i] is not None:
                        self.con_ineq_ub_idx.append(i)
                        self.con_ineq_ub_val.append(cons_ub[i])
            # END FOR (adding inequality)
        # END IF (constraints setup)

        # problem dimensions
        self.nx = copy.deepcopy(nx)   # number of variables
        self.nh = len(self.con_eq_idx)   # number of equality constraints
        self.ng = len(self.con_ineq_lb_idx) + len(self.con_ineq_ub_idx) + len(self.x_lb_idx) + len(self.x_ub_idx)   # number of inequality constraints
        self.ng_nonl = len(self.con_ineq_lb_idx) + len(self.con_ineq_ub_idx)   # number of nonlinear inequlities. NOTE/TODO (FFR): assumes that given constraints are all nonlinear
        # elastic problem dimensions
        self.nx_e = nx + 2 * self.nh + self.ng_nonl   # number of variables in elastic mode
        # NOTE: nh and ng_nonl remains the same in elastic mode
        
        # if necessary, we need to add a setup dummy constraint in the constraint function call (because the algorithm needs at least one equality and one inequality)
        self.flag_inequality_only = False
        self.flag_equality_only = False
        if self.nh == 0:
            print('\n*** No equality constraints. A dummy equality will be added in the function call.')
            print('WARNING: dummy constraint not implemented for elastic mode')
            self.flag_inequality_only = True
            self.nh = 1
        if self.ng == 0:
            print('\n*** No inequality constraints nor bounds. A dummy inequality will be added in the function call.')
            print('WARNING: dummy constraint not implemented for elastic mode')
            self.flag_equality_only = True
            self.ng = 1
        # TODO: dummy constraint + elastic mode. need to change nh and ng here?

        print('\n--- number of constraints (including dummy and bounds, all one-sided) ---')
        print('equality   h:', self.nh)
        print('inequality g:', self.ng, '| breakdown:', len(self.con_ineq_lb_idx), 'con LBs,', len(self.con_ineq_ub_idx), 'con UBs,', len(self.x_lb_idx), 'var LBs,', len(self.x_ub_idx), 'var UBs\n')
       
        # initialize penalty parameters for merit function. This will be updated during optimization
        self.rho = np.zeros(self.nh + self.ng_nonl)  # different penality value for each nonlinear constraint. rho = [rho_h, rho_g_nonl]
        self.delta_rho = 1.   # damping factor for penality parameter update
        self.rho_increase_count = 0
        self.rho_decrease_count = 0

        # initialize caches (to avoid duplicated calls on functions and gradient/Jacobian)
        self.f_cache = {'x': np.ones(nx) * 1e100, 'f': 0.}
        self.f_grad_cache = {'x': np.ones(nx) * 1e100, 'df/dx': np.zeros(nx)}
        self.con_cache = {'x': np.ones(nx) * 1e100, 'c': np.zeros(nc), 'h': np.zeros(self.nh), 'g': np.zeros(self.ng)}
        self.con_Jac_cache = {'x': np.ones(nx) * 1e100, 'dc/dx': np.zeros((nc, nx)), 'dh/dx': np.zeros((self.nh, nx)), 'dg/dx': np.zeros((self.ng, nx))}

        # initialize log for line search (reset at every major iterations)
        self.line_search_log = []
        
        # initialize functin call counters
        self.n_calls_f = 0
        self.n_calls_f_grad = 0
        self.n_calls_con = 0
        self.n_calls_con_Jac = 0

        # initialize search history
        self.f_hist = []   # all objective function calls
        self.f_grad_hist = []   # all objective gradient calls
        self.con_hist = []   # all constraint function calls
        self.con_Jac_hist = []   # all constraint Jacobian calls
        self.major_hist = []   # major iteration history

        # initilize exit status
        self.exit_status = -1

    def set_options(self, options_user={}):
        """
        Setup optimization options.

        Parameters
        ----------
        options : dict
            Dictionary of options.
        """

        # --- default options ---
        self.options = {}
        self.options['tol_opt'] = 1e-6
        self.options['tol_feas'] = 1e-6
        self.options['max_iter'] = 1000   # max major iterations
        self.options['major_step_limit'] = 10.0   # same definition as SNOPT

        # Hessian approximation
        self.options['hessian_reset_freq'] = 100   # reset Hessian every this many iterations
        self.options['hessian_definite_tol'] = 1e-8   # tolerance for checking positive definiteness of Hessian approximation
        self.options['hessian_initial_scaler'] = 1.0   # apply this scaling factor to H = I at iteration 0.  # TODO: scale this with |g|?
        self.options['hessian_correction_for_identity'] = False   # If True, apply the scaler to H = I based on the curvature information after taking the step and before the next BFGS update (Eq. 6.20 of Nocedal)
        
        # line search
        self.options['ls_scipy_c1'] = 1e-4       # scipy line search parameter c1 (sufficient decrease)
        self.options['ls_scipy_c2'] = 0.9        # scipy line search parameter c2 (curvature condition)
        self.options['ls_scipy_maxiter'] = 5     # scipy line search max iterations
        self.options['ls_backtrack_shrink_ratio'] = 0.3   # shrink factor for backtracking line search
        self.options['ls_backtrack_maxiter'] = 10         # max iterations for backtracking line search
        self.options['ls_backtrack_c1'] = 1e-5            # sufficient decrease parameter for backtracking line search

        # elastic mode
        self.options['elastic_weight'] = 1.e5   # initial elastic weight gamma
        
        # number of threads to be used for QP solver
        self.options['num_threads'] = 1
        # --- end of default options ---

        # override default options
        for key in options_user:
            if key not in self.options.keys():
                print('WARNING: option', key, 'is ignored')
            else:
                print('Option:', key, '->', options_user[key])
                self.options[key] = options_user[key]
        # END FOR
        print('\n')
   
    def _compute_optimality(self, x, dfdx, dLdx, lam, sig):
        """
        Compute optimality measure (same as SNOPT)

        Parameters
        ----------
        x : ndarray, shape (nx,)
            Optimization variables
        dfdx : ndarray, shape (nx,)
            Gradient of objective, used for normalization.
        dLdx : ndarray, shape (nx,)
            Gradient of Lagrangian w.r.t. x
        lam : ndarray, shape (nh,)
            Lagrange multipliers for equality constraints
        sig : ndarray, shape (ng,)
            Lagrange multipliers for inequality constraints

        Returns
        -------
        maxcomp : float
            Optimality measure
        """
        # TODO: check the complemenraty slackness (s * c) and multipliers sign (sig > 0) like SNOPT does.
        # TODO: make this optimality measure consistent with SNOPT

        # normalization (same as SNOPT)
        gNorm = np.linalg.norm(dfdx, ord=1) / np.sqrt(len(dfdx))   # approx. 2-norm of objective gradient
        lag_multi_nonl = np.concatenate((lam, sig[:self.ng_nonl]))   # all nonlinear lagrange multipliers = all multipliers except for bound constraints (Note that I currenly don't support linear non-bound constraints yet). TODO: change this if I support linear non-bound constraints. SNOPT uses non-bound linear multipliers here.
        if len(lag_multi_nonl) > 0:
            piNorm = max(np.linalg.norm(lag_multi_nonl, ord=np.inf), 1)
        else:
            piNorm = 1.
        # END IF
        normalizer = max(gNorm, piNorm)
        maxcomp = np.linalg.norm(dLdx, ord=np.inf) / normalizer

        # also compute the 2-norm
        opt_l2 = np.linalg.norm(dLdx, ord=2) / normalizer

        return maxcomp, opt_l2

    # ---------------------------------------------
    #  function wrappers for the normal-mode NLP
    # ---------------------------------------------
    def _obj(self, x):
        """ objective function wrapper """
        if not np.array_equal(x, self.f_cache['x']):
            # new point; evaluate function
            self.f_cache['x'] = x
            self.f_cache['f'] = self._obj_func(x)
            # cache and save search history
            self.n_calls_f += 1
            self.f_hist.append(copy.deepcopy(self.f_cache))
        # otherwise, return cached function value
        return self.f_cache['f']

    def _obj_grad(self, x):
        """ objective gradient wrapper """
        if not np.array_equal(x, self.f_grad_cache['x']):
            # new point; evaluate gradient
            self.f_grad_cache['x'] = x
            self.f_grad_cache['df/dx'] = self._obj_grad_func(x)
            # cache and save search history
            self.n_calls_f_grad += 1
            self.f_grad_hist.append(copy.deepcopy(self.f_grad_cache))
        return self.f_grad_cache['df/dx']

    def _cons(self, x):
        """ constraint function wrapper """
        if not np.array_equal(x, self.con_cache['x']):
            # new point; evaluate function
            self.con_cache['x'] = x
            c = self._cons_func(x)
            self.con_cache['c'] = c

            # equality constraints h = 0
            self.con_cache['h'] = c[self.con_eq_idx] - self.con_eq_val

            # inequality constraints g <= 0 (including variable bounds)
            g_lb = self.con_ineq_lb_val - c[self.con_ineq_lb_idx]
            g_ub = c[self.con_ineq_ub_idx] - self.con_ineq_ub_val
            x_lb = self.x_lb_val - x[self.x_lb_idx]
            x_ub = x[self.x_ub_idx] - self.x_ub_val
            self.con_cache['g'] = np.concatenate((g_lb, g_ub, x_lb, x_ub))

            # add dummy constraints if necessary
            if self.flag_inequality_only:
                self.con_cache['h'] = np.concatenate((self.con_cache['h'], np.array([0.])))
            if self.flag_equality_only:
                self.con_cache['g'] = np.concatenate((self.con_cache['g'], np.array([-1.])))

            # cache and save search history
            self.n_calls_con += 1
            self.con_hist.append(copy.deepcopy(self.con_cache))
        # END IF (cache)
        return self.con_cache['h'], self.con_cache['g']

    def _cons_jac(self, x):
        """ constraint Jacobian wrapper """
        if not np.array_equal(x, self.con_Jac_cache['x']):
            # new point; evaluate gradient
            self.con_Jac_cache['x'] = x
            dcdx = self._jac_cons_func(x)
            self.con_Jac_cache['dc/dx'] = dcdx

            # Jacobian of equalities
            self.con_Jac_cache['dh/dx'] = dcdx[self.con_eq_idx, :]

            # Jacobian of inequalities
            jac_g_lb = -dcdx[self.con_ineq_lb_idx, :]
            jac_g_ub = dcdx[self.con_ineq_ub_idx, :]
            self.con_Jac_cache['dg/dx'] = np.concatenate((jac_g_lb, jac_g_ub, self.jac_x_lb, self.jac_x_ub), axis=0)

            # add dummy constraints if necessary
            if self.flag_inequality_only:
                self.con_Jac_cache['dh/dx'] = np.concatenate((self.con_Jac_cache['dh/dx'], np.zeros((1, self.nx))), axis=0).reshape((self.nh, self.nx))
            if self.flag_equality_only:
                self.con_Jac_cache['dg/dx'] = np.concatenate((self.con_Jac_cache['dg/dx'], np.zeros((1, self.nx))), axis=0).reshape((self.ng, self.nx))

            # cache and save search history
            self.n_calls_con_Jac += 1
            self.con_Jac_hist.append(copy.deepcopy(self.con_Jac_cache))
        # END IF (cache)
        return self.con_Jac_cache['dh/dx'], self.con_Jac_cache['dg/dx']

    # ------------------------------------
    #  merit functions for line search
    # ------------------------------------
    def _unpack_linesearh_var(self, v):
        """
        Unpack line search variable v = [x, lam, sig_nonl, slack_nonl]
        """
        x = v[:self.nx]
        lam = v[self.nx : self.nx + self.nh]
        sig = v[self.nx + self.nh : self.nx + self.nh + self.ng_nonl]
        slack = v[self.nx + self.nh + self.ng_nonl:]
        
        return x, lam, sig, slack
        
    def _merit_func(self, v):
        """
        Augmented Lagrangian metit function for line search.

        Parameters
        ----------
        v : ndarray, shape (nx + nh + 2 * ng_nonl,)
            Optimization variables & Lagrange multipliers & slack variables, vars = [x, lam, sig_nonl, slack_nonl]
        
        Returns
        -------
        phi: float
            Merit function value
        """

        x, lam, sig, slack = self._unpack_linesearh_var(v)

        f = self._obj(x)
        h, g = self._cons(x)
        rho_h = self.rho[:self.nh]
        rho_g = self.rho[self.nh:]   # this is only for g_nonl

        # merit function includes nonlinear constraints only. equality h is (assumed to be) all nonlinear
        g = g[:self.ng_nonl]

        phi = f + np.dot(lam, h) + np.dot(sig, g + slack) + 0.5 * (np.dot(rho_h * h, h) + np.dot(rho_g * (g + slack), g + slack))

        # log merit function value
        self.line_search_log.append({'v': v, 'phi': phi})

        return phi

    def _merit_func_grad(self, v):
        """
        Gradient of augmented Lagrangian metit function.

        Parameters
        ----------
        v : ndarray, shape (nx + nh + 2 * ng_nonl,)
            Optimization variables & Lagrange multipliers & slack variables, vars = [x, lam, sig_nonl, slack_nonl]
        
        Returns
        -------
        dphi_dv : ndarray, shape (nx + nh + 2 * ng_nonl,)
            Gradient of merit function w.r.t. v
        """

        x, lam, sig, slack = self._unpack_linesearh_var(v)

        h, g = self._cons(x)
        dfdx = self._obj_grad(x)
        dhdx, dgdx = self._cons_jac(x)
        rho_h = self.rho[:self.nh]
        rho_g = self.rho[self.nh:]

        g = g[:self.ng_nonl]
        dgdx = dgdx[:self.ng_nonl, :]

        dphi_dx = dfdx + dhdx.T @ lam + dgdx.T @ sig + dhdx.T @ (rho_h * h) + dgdx.T @ ((g + slack) * rho_g)  # shape (nx)
        dphi_dslack = sig + rho_g * (g + slack)
        # here, dphi/dv = [dphi/dx, dphi/dlam, dphi/dsig, dphi/dslack], where dphi/dlam = h.T, dphi/dsig = (g + s).T
        dphi_dv = np.concatenate((dphi_dx, h, (g + slack), dphi_dslack))  # shape (nx + nh + 2 * ng)

        return dphi_dv

    def __merit_func_derivatives_check(self, v):
        """
        Verify derivatives of merit function by FD; not to be used by the SQP algorithm.
        """

        dmdv = self._merit_func_grad(v)

        # sanity check on the shape of v and dmdv
        if len(v) != self.nx + self.nh + 2 * self.ng_nonl:
            raise RuntimeError('v has wrong shape')
        if len(dmdv) != self.nx + self.nh + 2 * self.ng_nonl:
            raise RuntimeError('dmerit/dv has wrong shape')

        # finite difference
        dmdv_fd = np.zeros_like(dmdv)
        eps = 1e-8
        for i in range(len(v)):
            dv = np.zeros_like(v)
            dv[i] = eps
            dmdv_fd[i] = (self._merit_func(v + dv) - self._merit_func(v)) / eps
        # END FOR

        print('\n ****** merit function derivatives check *****')
        print('dmdv:', dmdv)
        print('dmdv_fd:', dmdv_fd)
        print('merit function derivatives rel errors:', np.max(abs(dmdv - dmdv_fd) / np.maximum(abs(dmdv), 1e-10)))
        print('*********************************************')

    # ------------------------------------
    # components of SQP algorithm
    # ------------------------------------
    def _damped_BFGS(self, H, s, y):
        """
        Damped BFGS update of Hessian approximation.
        See Eqs. (5.91)-(5.94) in pp.199-200 of Engineering Design Optimization (2021).
        
        Parameters
        ----------
        H : ndarray, shape (nx, nx)
            Current Hessian approximation.
            In elastic mode, we only update the Hessian entries for nonlinear variables x, thus shape is always (nx, nx)
        s : ndarray, shape (nx,)
            Step vector (x_{k+1} - x_k)
        y : ndarray, shape (nx,)
            Gradient difference vector (dL/dx_{k+1} - dL/dx_k)
            
        Returns
        -------
        H_next: ndarray, shape (nx, nx)
            New Hessian approximation
        """

        # reshape to column vectors
        s = s.reshape((self.nx, 1))
        y = y.reshape((self.nx, 1))
        
        Hs = H @ s

        # damping
        if s.T @ y >= 0.2 * s.T @ Hs:
            theta = 1.0
        else:
            theta = 0.8 * s.T @ Hs / (s.T @ Hs - s.T @ y)

        r = theta * y + (1.0 - theta) * Hs  # shape (nx, 1)

        # Hessian update with damping
        H_next = H - Hs @ s.T @ H / (s.T @ Hs) + r @ r.T / (r.T @ s)

        return H_next

    def _solve_QP(self, H, dfdx, dhdx, h, dgdx, g, lam, sig, slack):
        """
        Solve QP subproblem.
        
        Parameters
        ----------
        H : ndarray, shape (nx, nx)
            Current Hessian approximation.
        dfdx : ndarray, shape (nx)
            Gradient of objective
        dhdx : ndarray, shape (nh, nx)
            Jacobian of equality constraints (h=0)
        h : ndarray, shape (nh,)
            Current value of equality constraints
        dgdx : ndarray, shape (ng, nx)
            Jacobian of inequality constraints (g<=0)
        g : ndarray, shape (ng)
            Current value of inequality constraints
        lam : ndarray, shape (nh,)
            Current estimate of the Lagrange multipliers of equality constraints
        sig : ndarray, shape (ng)
            Current estimate of the Lagrange multipliers of inequality constraints
        slack : ndarray, shape (ng)
            Slack variables for inequality constraints
            
        Returns
        -------
        p_x: ndarray, shape (nx)
            Step vector for design variables
        p_lam: ndarray, shape (nh)
            Step vector for Lagrange multipliers of equality constraints
        p_sig : ndarray, shape (ng)
            Step vector for Lagrange multipliers of inequality constraints
        p_slack : ndarray, shape (ng)
            Step vector for slack variables
        """

        # get problem dimensions. nx and ng changes depending on the NLP mode (normal or elastic). nh remains unchanged
        if self.elastic_mode:
            nx = self.nx_e
        else:
            nx = self.nx
        # END IF
        
        if self.elastic_mode:
            # solve QP corresponding to elastic problem, which should be always feasible.
            # augment Hessian with 0s (because elastic variables are linear and their 2nd derivs are 0)
            H_elastic = np.zeros((self.nx_e, self.nx_e))
            H_elastic[:self.nx, :self.nx] = H
            H = H_elastic
            # augment dfdx, dhdx ,dgdx for augmented variable x := [x, v, w, z]
            dfdv = self.gamma * np.ones(self.nh)
            dfdw = self.gamma * np.ones(self.nh)
            dfdz = self.gamma * np.ones(self.ng_nonl)
            dfdx = np.concatenate((dfdx, dfdv, dfdw, dfdz))   # shape (nx_e)

            dhdx = np.concatenate((dhdx, -np.eye(self.nh), np.eye(self.nh), np.zeros((self.nh, self.ng_nonl))), axis=1)   # [dh/dx, dh/dv=eye, dh/dw=eye, dh/dz=0]
    
            g = np.concatenate((g, np.zeros(self.nh * 2 + self.ng_nonl)))   # shape (ng_e)
            dgdx_nonl = dgdx[:self.ng_nonl, :]
            dgdx_lin = dgdx[self.ng_nonl:, :]
            dgdx_nonl_elastic = np.concatenate((dgdx_nonl, np.zeros((self.ng_nonl, self.nh)), np.zeros((self.ng_nonl, self.nh)), -np.eye(self.ng_nonl)), axis=1)   # [dg_nonl/dx, dg_nonl/dv=0, dg_nonl/dw=0, dg_nonl/dz]
            dgdx_lin_elastic = np.concatenate((dgdx_lin, np.zeros((self.ng - self.ng_nonl, 2 * self.nh + self.ng_nonl))), axis=1)   # [dg_lin/dx, dg_lin/dv=0, dg_lin/dw=0, dg_lin/dz=0]
            # for bounds on v, w, z
            dgdx_v_elastic = np.concatenate((np.zeros((self.nh, self.nx)), -np.eye(self.nh), np.zeros((self.nh, self.nh)), np.zeros((self.nh, self.ng_nonl))), axis=1)   # [dx=0, dgdv, dw=0, dz=0]
            dgdx_w_elastic = np.concatenate((np.zeros((self.nh, self.nx)), np.zeros((self.nh, self.nh)), -np.eye(self.nh), np.zeros((self.nh, self.ng_nonl))), axis=1)   # [dx=0, dv=0, dgdw, dz=0]
            dgdx_z_elastic = np.concatenate((np.zeros((self.ng_nonl, self.nx)), np.zeros((self.ng_nonl, self.nh)), np.zeros((self.ng_nonl, self.nh)), -np.eye(self.ng_nonl)), axis=1)   # [dx=0, dv=0, dw=0, dgdz]

            dgdx = np.concatenate((dgdx_nonl_elastic, dgdx_lin_elastic, dgdx_v_elastic, dgdx_w_elastic, dgdx_z_elastic), axis=0)
        # END IF (elastic QP formulation)

        if self.flag_unconstrained:
            # unconstrained problem. Just take a quasi-Newton step.
            # don't update the Lagrange multipliers (should remain lam=0)
            p_x = np.linalg.solve(H, -dfdx)
            p_lam = np.zeros(self.nh)
            p_sig = np.zeros(self.ng)
            p_slack = np.zeros(self.ng)

            if self.elastic_mode:
                # we should never enter the elastic mode for unconstrained problem, so something is wrong
                raise RuntimeError('Entered elastic mode for unconstrained problem. Something is wrong with the algorithm.')

        elif self.flag_equality_only:
            if self.elastic_mode:
                print('WARNING: equality-only problem in elastic mode might have a bug.')
            # equality-constrained problem (no inequality). Solving the QP for equality constraints reduces to solving a linear KKT system
            p_sig = np.zeros(self.ng)   # Lagrange multipliers for inequality constraints doesn't matter
            p_slack = np.zeros(self.ng)

            dLdx = dfdx + np.dot(dhdx.T, lam) + np.dot(dgdx.T, sig)

            # QP with only equality constraints (no inequalities) can be solved by the following linear system
            # See Eq. (5.62) in pp.187 of Engineering Design Optimization (2021).
            A = np.concatenate((np.concatenate((H, dhdx.T), axis=1), np.concatenate((dhdx, np.zeros((self.nh, self.nh))), axis=1)), axis=0)  # shape (nx+nh, nx+nh)
            b = np.concatenate((-dLdx, -h), axis=0)  # shape (nx+nh,)
            p = np.linalg.solve(A, b)

            p_x = p[:nx][:self.nx]
            p_lam = p[nx:]

        else:
            # inequality-constrained problem. Solve the QP iteratively via CVXPY

            # formulate QP
            x = cvxpy.Variable(nx)
            obj = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, H, assume_PSD=True) + dfdx.T @ x)
            cons = [dgdx @ x <= -g,
                    dhdx @ x == -h]
            if self.flag_inequality_only or self.flag_bounds_only:
                # remove equality from the QP
                cons.pop(1)

            # TODO: set initial guess
            qp = cvxpy.Problem(obj, cons)
            mosek_params = {'MSK_IPAR_LOG': 0,   # disable print
                            'MSK_IPAR_MAX_NUM_WARNINGS': 0,   # disable warnings
                            'MSK_DPAR_SEMIDEFINITE_TOL_APPROX': 10000.,   # disable Hessian definiteness check in Mosek (instead we check at SQP level)
                            'MSK_DPAR_INTPNT_QO_TOL_DFEAS': 1e-10,   # dual feasibility (optimality) tolerance
                            'MSK_DPAR_INTPNT_QO_TOL_MU_RED': 1e-10,   # complementarity gap tolerance
                            'MSK_DPAR_INTPNT_QO_TOL_PFEAS': 1e-10,   # primal feasibility tolerance
                            }
            try:
                ### qp.solve(solver='CVXOPT', verbose=False, max_iters=1000, abstol=1e-8, reltol=1e-8, feastol=1e-8)
                ### qp.solve(solver='MOSEK', verbose=False, mosek_params=mosek_params)
                qp.solve(solver='GUROBI', verbose=False, Threads=self.options['num_threads'])
            except cvxpy.error.SolverError:
                print('   - WARNING: QP solver failed. (cvxpy.error.SolverError)')
                return None, None, None, None, False
            if qp.status != 'optimal':
                print('   - WARNING: QP solver failed (qp.status != optimal).')
                ### print('QP sol =', sol)
                return None, None, None, None, False

            # get QP solution (which gives search directions)
            p_x = x.value[:self.nx]

            if self.flag_inequality_only or self.flag_bounds_only:
                p_lam = np.zeros(self.nh)
            else:
                p_lam = qp.constraints[1].dual_value - lam
            
            if self.flag_equality_only:
                p_sig = np.zeros(self.ng)
                p_slack = np.zeros(self.ng)
            else:
                p_sig = qp.constraints[0].dual_value[:self.ng] - sig
                # compute slack
                slack_new = -(dgdx[:self.ng, :self.nx] @ p_x + g[:self.ng])
                p_slack = slack_new[:self.ng] - slack

            # print('--- CVXPY ---')
            # print('p_x', p_x)
            # print('p_lam', p_lam)
            # print('p_sig', p_sig)
            # print('p_slack', p_slack)
        # END IF

        return p_x, p_lam, p_sig, p_slack, True

    def _line_search(self, x, p_x, lam, p_lam, sig, p_sig, slack, p_slack, H, backtrack=False):
        """
        Line search on a merit function.
        We use the augmented Lagrangian function as a merit function, and the step size applies to both the variable x and Lagrange multiplier lam.

        Parameters
        ----------
        x : ndarray, shape (nx)
            Current design variables
        p_x : ndarray, shape (nx)
            Search direction of design variables
        lam : ndarray, shape (nh,)
            Current Lagrange multipliers for equality constraints at x
        p_lam : ndarray, shape (nh,)
            Search direction of Lagrange multipliers lam
        sig : ndarray, shape (ng)
            Current Lagrange multipliers for inequality constraints at x
        p_sig : ndarray, shape (ng)
            Search direction of Lagrange multipliers sig
        slack : ndarray, shape (ng)
            Current slack variables for inequality constraints at x
        p_slack : ndarray, shape (ng)
            Search direction of slack variables
        H : ndarray, shape (nx, nx)
            Current Hessian approximation, to be used to update the penalty parameters if necessary
        backtrack : bool, optional
            If True, skip the Scipy's line search, and just do a simple backtracking with sufficient decrease condition

        Returns
        -------
        alpha: float
            Step size
        """

        # form line search variables v := [x, lam, sig_nonl, slack_nonl] and p_v := [p_x, p_lam, p_sig_nonl, p_slack_nonl]
        v0 = np.concatenate((x, lam, sig[:self.ng_nonl], slack[:self.ng_nonl]))
        p_v = np.concatenate((p_x, p_lam, p_sig[:self.ng_nonl], p_slack[:self.ng_nonl]))

        ### line search derivatives check
        ### self.__merit_func_derivatives_check(v0)

        """
        # update a scaler penalty parameter (Gill 1986, Some Theoretical Properties of an Augmented Lagrangian Merit Function, Eqs. (4.15)--(4.16))
        if self.rho[0] > 10:
            self.rho = 0.5 * self.rho   # try reducing the penalty parameter.

        phi_prime_0 = np.dot(self._merit_func_grad(v0), p_v)   # directional derivatives of merit function
        if phi_prime_0 > -0.5 * np.dot(p_x, H @ p_x):
            h, g = self._cons(x)
            xi = np.concatenate((p_lam, p_sig))
            denom = np.concatenate((h, g + slack))
            mu_hat = 2 * np.linalg.norm(xi) / np.maximum(np.linalg.norm(denom), 1e-10)    # avoid 0 division
            self.rho = max(mu_hat, 2 * self.rho[0]) * np.ones_like(self.rho)
            # update phi_prime_0 with the new penalty parameter
            phi_prime_0 = np.dot(self._merit_func_grad(v0), p_v)
        # """

        # --- update a vector of penality parameters (Eldersveld 1992, PhD thesis) ---
        h, g = self._cons(x)

        # use only nonlinear constraints
        g = g[:self.ng_nonl]
        sig = sig[:self.ng_nonl]
        p_sig = p_sig[:self.ng_nonl]
        slack = slack[:self.ng_nonl]
        p_slack = p_slack[:self.ng_nonl]

        cons = np.concatenate((h, g + slack))
        r = cons ** 2
        if np.max(r) > 1e-15:
            # if max(r) = 0, then all constraints are satisfied and the merit function is independent of rho at the current point. No need to update rho.
            # otherwise, we may need to increase the penalty parameter.
            
            lam_current = np.concatenate((lam, sig))   # Lagrange multipliers for nonlinear constraint at current point. This appears as "lambda" in Eldersveld1992
            lam_qp = lam_current + np.concatenate((p_lam, p_sig))   # Lagrange multipliers for nonlinear at QP solution. This appears as "mu" in Eldersveld1992
            theta = np.dot(self._obj_grad(x), p_x) - np.dot((2 * lam_current - lam_qp), cons) + 0.5 * np.dot(p_x, H @ p_x)   # Eldersveld 1992.
            ### theta = -0.5 * np.dot(p_x, H @ p_x) + np.dot(p_slack, sig + p_sig) + 2 * np.dot(cons, np.concatenate((p_lam, p_sig)))   # Gill 1986   NOTE: seems something is wrong here. Eldersveld 1992 and Gill1986 should match here, but they do not...
            theta = max(theta, 1e-10)   # theta should be positive
            rho_star = theta / np.dot(r, r) * r

            # decrease penalty parameter (Gill 2005, pp 105)
            rho_hat = (self.rho * (rho_star + self.delta_rho))**0.5
            rho_hat[self.rho < 4 * (rho_star + self.delta_rho)] = self.rho[self.rho < 4 * (rho_star + self.delta_rho)]
            rho_new = np.maximum(rho_star, rho_hat)

            # update damping parameter
            if self.delta_rho < 100:   # upper bound
                if np.linalg.norm(rho_new) > np.linalg.norm(self.rho):
                    if self.rho_decrease_count >= 2:
                        self.delta_rho *= 2.   # norm(rho) increased after a consecutive sequence of iterations in which the penalty norm decreased
                        ### print('   - Increased delta_rho (case A), delta_rho =', self.delta_rho)
                    self.rho_decrease_count = 0
                    self.rho_increase_count += 1
                        
                elif np.linalg.norm(rho_new) < np.linalg.norm(self.rho):
                    if self.rho_increase_count >= 2:
                        self.delta_rho *= 2.0   # norm(rho) decreased after a consecutive sequence of iterations in which the penalty norm increased
                        ### print('   - Increased delta_rho (case B), delta_rho =', self.delta_rho)
                    self.rho_increase_count = 0
                    self.rho_decrease_count += 1
            # END IF

            # sanity check - phi_prime_0 verification
            # phi_prime_0 = np.dot(self._merit_func_grad(v0), p_v)
            # phi_prime_0_gill = -np.dot(p_x, H @ p_x) + np.dot(p_slack, sig + p_sig) + 2 * np.dot(cons, np.concatenate((p_lam, p_sig))) - np.dot(cons * self.rho, cons)   # NOTE: seems wrong here.
            # phi_prime_0_elda = np.dot(self._obj_grad(x), p_x) - np.dot((2 * lam_current - lam_qp), cons) - np.dot(self.rho * cons, cons)
            # if abs(phi_prime_0 - phi_prime_0_elda) > 1e-5:
            #     print('   - WARNING: phi_prime seems wrong!')
            #     print('     phi_prime (eval):', phi_prime_0)
            #     print('     phi_prime (Gill):', phi_prime_0_gill)
            #     print('     phi_prime (Elde):', phi_prime_0_elda)
            #     print('     error =', phi_prime_0 - phi_prime_0_elda)
            #     # print('current rho =', self.rho)
            #     # print('new rho =', rho_new)

            self.rho = rho_new * 1.
        # END IF (max(r) > 0)

        # directional derivative of the merit function with the updated penalty parameters
        phi_prime_0 = np.dot(self._merit_func_grad(v0), p_v)
        # --- end of penalty parameter update ---

        # sanity check (1): if the condition "phi_prime_0 <= -0.5 * np.dot(p_x, H @ p_x)" is satisfied
        # NOTE/TODO: this manual rho update does not necessalily help?
        pHp = np.dot(p_x, H @ p_x)
        if phi_prime_0 - (-0.5 * pHp) >= 1e-5 * abs(0.5 * pHp):
            print('   - WARNING: The condition for the merit function decrease was not satisfied.')
            # manually increase rho so that the condition is satisfied
            rho_add_factor = 0.001
            while rho_add_factor < 10000:
                rho_factor = 1. + rho_add_factor
                self.rho *= rho_factor
                phi_prime_0 = np.dot(self._merit_func_grad(v0), p_v)
                if phi_prime_0 - (-0.5 * pHp) <= 0:
                    print('     Manually adjusted rho to satisfy the condition by the factor =', rho_factor)
                    break
                else:
                    # the condition still not satisfied, increase rho
                    self.rho /= rho_factor
                    rho_add_factor *= 2.
            # END WHILE

        # sanity check (2): phi_prime_0 should be negative
        if phi_prime_0 > 0:
            print('   - WARNING: phi_prime_0 is positive! Line search will fail...')
            min_eval = np.min(np.linalg.eig(H)[0])
            print('     min eval of Hessian:', min_eval)

        # limit maximum step size
        beta_bar = self.options['major_step_limit'] * (1. + np.linalg.norm(x)) / np.linalg.norm(p_x)
        max_step_size = min(1.0, beta_bar)
        if max_step_size < 1.0:
            print('   - Set max step size to', max_step_size, 'in line search')

        # line search on both design variables and Lagrange multipliers
        self.line_search_log = []   # reset line search log

        # --- line search ---
        if backtrack:
            # just do a simple backtracking line search
            phi_0 = self._merit_func(v0)
            # do just backtracking line search
            maxiter = self.options['ls_backtrack_maxiter']
            shrink_ratio = self.options['ls_backtrack_shrink_ratio']
            alpha = 1.0
            for i in range(maxiter):
                merit_new = self._merit_func(v0 + alpha * p_v)
                if merit_new <= phi_0 + 1e-4 * alpha * phi_prime_0:
                    # sufficient descrease condition satisfied
                    return alpha, merit_new
                else:
                    alpha *= shrink_ratio

            print("   - Backtraking line search (as primary) failed.")
            return None, None

        else:
            # do fancy line search
            alpha, func_calls, grad_calls, merit_new, _, _, = scipy_line_search(f=self._merit_func, myfprime=self._merit_func_grad, xk=v0, pk=p_v, c1=self.options['ls_scipy_c1'], c2=self.options['ls_scipy_c2'], amax=max_step_size, maxiter=self.options['ls_scipy_maxiter'])
            # NOTE: don't set amax > 1. That would violate linear constraints as we don't penalize linear constraint violations in the merit function.
            ### print('line search func_calls: ', func_calls, 'grad_calls: ', grad_calls)

        if alpha is not None:
            # line search succeeded
            return alpha, merit_new
            
        else:
            # line search failed (likely because it could not satisfiy the curvature condition).   NOTE: in Lagrange merit function, the second term (lam.T h) can be <<0 when lam <<0. This can cause the curvature condition to fail.
            # check if a line-searched point can satisfy the sufficient decrease condition.
            print("   - Scipy's line search failed. Searching a point that satisfies the sufficient decrease condition.")
            phi_0 = self.line_search_log[0]['phi']   # line_search_log[0] is the current point v0.
            for i in range(1, len(self.line_search_log)):
                alpha_i = np.linalg.norm(self.line_search_log[i]['v'] - v0) / np.linalg.norm(p_v)
                if self.line_search_log[i]['phi'] <= phi_0 + self.options['ls_backtrack_c1'] * alpha_i * phi_prime_0:
                    # sufficient decrease condition satisfied
                    # print('found a point at i =', i)
                    return alpha_i, self.line_search_log[i]['phi']
            
            # If no point from the previous line search can satisfy the sufficient decrease condition, try a backtrack line search again from scratch with only the sufficient decrease condition
            print("     No point found. Trying a simple backtracking line search.")
            maxiter = self.options['ls_backtrack_maxiter']
            shrink_ratio = self.options['ls_backtrack_shrink_ratio']
            alpha = 1.0
            for i in range(maxiter):
                merit_new = self._merit_func(v0 + alpha * p_v)
                if merit_new <= phi_0 + self.options['ls_backtrack_c1'] * alpha * phi_prime_0:
                    # sufficient descrease condition satisfied
                    return alpha, merit_new
                else:
                    alpha *= shrink_ratio
            # END FOR (backtrack line search)

            # line search failed! plot merit function for debugging
            print("     Backtraking line search failed. Printing debug info...")
            
            # --- debug print ---
            print('******************************')
            print('ERROR: Line Search Failed!')
            print('    alpha =', alpha)
            print('    penalty rho =', np.linalg.norm(self.rho))
            print('    phi_prime_0 =', phi_prime_0)
            # print('    x =', x)
            # print('    p_x =', p_x)
            # print('lam =', lam)
            # print('p_lam =', p_lam)
            print('******************************')

            """
            # plot merit function along the search direction
            alpha_plot = np.linspace(0, 1.0, 30)
            merits = np.zeros(len(alpha_plot))
            merits_deriv = np.zeros(len(alpha_plot))
            for i in range(len(alpha_plot)):
                merits[i] = self._merit_func(v0 + alpha_plot[i] * p_v)
                merits_deriv[i] = np.dot(self._merit_func_grad(v0 + alpha_plot[i] * p_v), p_v)
            # END FOR
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(alpha_plot, merits)
            ax[0].set_ylabel('merit')
            ax[1].plot(alpha_plot, merits_deriv)
            ax[1].set_xlabel('alpha')
            ax[1].set_ylabel('merit derivative')
            plt.savefig('line_search_debug.png', bbox_inches='tight')
            """

            return None, None
            # END IF
        # END IF (line search failed)

    def _setup_elastic_mode(self):
        """
        Setup elastic mode.
        """

        # trigger elastic mode
        self.elastic_mode = True

        # reduce Hessian frequency for elastic mode (this usually works better)
        ### self.hessian_reset_freq = 10   # this is probably a bad idea???
        ### self.hessian_reset_freq = int(self.options['hessian_reset_freq'] / 2)
        print('   - Initiated elastic mode. Changed Hessian reset frequency to', self.hessian_reset_freq)

    def _compute_slack(self, g, sig):
        """ compute slack variables for inequality constraints """

        # update slack variables for the next iteration.
        slack = np.maximum(-g, 0.)
        # Here, nonlinear slacks are adjusted to minimize the merit function as a function of s, as described in Gill 1986, Eq. (2.8)
        rho_g_nonl = self.rho[self.nh:]
        rho_non0_idx = rho_g_nonl > 0
        slack[:self.ng_nonl][rho_non0_idx] = np.maximum(-g[:self.ng_nonl][rho_non0_idx] - sig[:self.ng_nonl][rho_non0_idx] / rho_g_nonl[rho_non0_idx > 0], 0.)

        return slack

    def _reset_Hessian(self, mode, H=None, current_point=None, prev_point=None):
        """
        Reset Hessian matrix to identity, scaled identity, or diagonal

        Parameters
        ----------
        mode : str
            Hessian reset mode. 'diag', 'I', or 'I_scaled'
        H : ndarray, shape (nx, nx)
            Current Hessian matrix
        current_point : dict
            Current point. Keys = 'x', 'lam', 'sig', 'dLdx'
        prev_point : dict
            Previous point. Keys = 'x', 'dfdx', 'dhdx', 'dgdx'
        """

        if mode == 'diag':
            # reset Hessian to diagonal matrix
            H = np.diag(np.diag(H))
            print('   - Hessian reset to diagonal matrix')
        elif mode == 'I':
            # reset Hessian to identity matrix
            H = np.eye(self.nx)
            print('   - Hessian reset to identity matrix')
        elif mode == 'I_scaled':
            # reset Hessian to scaled identity matrix
            x = current_point['x']
            lam = current_point['lam']
            sig = current_point['sig']
            dLdx = current_point['dLdx']
            x_prev = prev_point['x']
            dfdx_prev = prev_point['dfdx']
            dhdx_prev = prev_point['dhdx']
            dgdx_prev = prev_point['dgdx']
            
            s = x - x_prev
            dLdx_prev = dfdx_prev + np.dot(dhdx_prev.T, lam) + np.dot(dgdx_prev.T, sig)
            y = dLdx - dLdx_prev
            scaler = np.dot(s, y) / np.dot(y, y)
            if scaler <= 0:
                # scaled identity is negative definite. Reset to identity instead.
                scaler = 1.
            # END IF
            H = np.eye(self.nx) * scaler
            print('   - Hessian reset to scaled identity with scalar =', scaler)
        else:
            raise ValueError('Invalid Hessian reset mode:', mode)

        return H

    def optimize(self, x0):
        """
        SQP main algorithm.
        
        Parameters
        ----------
        x0 : ndarray, shape (nx,)
            Initial guess of design variables
            
        Returns
        -------
        major_hist : list of dicts
            Major iteration history. Each entry is a dict that stores x, f, h at each major iterations. keys = ['x', 'f', 'h']
        f_hist : list of dicts
            Function evaluation history, including line search iterations. Each entry is a dict that stores x, f at each function calls. keys = ['x', 'f']
        """

        # --- initial evaluation ---
        # evaluate functions and gradients/Jacobians at x0
        x = x0 * 1.  # make a copy, also convert to float
        f = self._obj(x)
        dfdx = self._obj_grad(x)  # shape (nh)
        h, g = self._cons(x)  # shape (nh), (ng)
        dhdx, dgdx = self._cons_jac(x)  # shape (nh, nx), (ng, nx)

        # --- other initialization ---
        # initial Lagrange multipliers
        lam = np.zeros(self.nh)   # for equalities
        sig = np.zeros(self.ng)   # for inequalities

        # compute nonlinear slack (Gill 1986, Eq. 2.8). Slack will be used only in the line search, and this replaces g <= 0 with g + s = 0 with s >= 0.
        slack = self._compute_slack(g, sig)

        # gradient of Lagrangian
        dLdx = dfdx + np.dot(dhdx.T, lam) + np.dot(dgdx.T, sig)  # shape (nx)

        # initial optimality and feasibility
        optimality, opt_l2 = self._compute_optimality(x, dfdx, dLdx, lam, sig)
        con_violation = np.concatenate((h, np.maximum(g, 0.)))
        feasibility = np.linalg.norm(con_violation, ord=np.inf) / max(1, np.linalg.norm(x, ord=np.inf))

        # log major iteration history
        self.major_hist.append({'x': x, 'f': f, 'h': h, 'g': g, 'flag_elastic': self.elastic_mode, 'opt_inf': optimality, 'opt_l2': opt_l2, 'feas': feasibility})

        # initialize the previous x and gradients (used for BFGS update)
        x_prev = np.zeros_like(x)
        dfdx_prev = np.zeros_like(dfdx)
        dhdx_prev = np.zeros_like(dhdx)
        dgdx_prev = np.zeros_like(dgdx)

        # iteration counter
        k = 0

        # header for iteration print. Later I can use this to print iteration type
        iter_print = '   '

        # Hessian reset frequency
        self.hessian_reset_freq = self.options['hessian_reset_freq']

        # initial elastic weight
        self.gamma = self.options['elastic_weight']

        # print initial evaluation. initial merit value = objective function value because penalty = 0.
        print('Iter      Step     Feasible  Optimal   Merit         Penalty')
        print(' %3d %s   %1.1E  %1.2E  %1.2E  %+1.5E  %1.2E' % (0, ' - ', 0., feasibility, optimality, f, np.linalg.norm(self.rho)))

        # --- SQP main loop ---
        while k < self.options['max_iter'] and (optimality > self.options['tol_opt'] or feasibility > self.options['tol_feas']):

            # --- Hessian approximation ---
            if k == 0 or k % self.hessian_reset_freq == 0:
                if k == 0 or self.elastic_mode:
                    # initialize Hessian to identity
                    H = self._reset_Hessian(mode='I')
                    if k == 0:
                        # apply initial scaling factor
                        H *= self.options['hessian_initial_scaler']
                elif self.elastic_mode:
                    # reset to self-scaled identity
                    H = self._reset_Hessian(mode='I_scaled', current_point={'x': x, 'dLdx': dLdx, 'lam': lam, 'sig': sig}, prev_point={'x': x_prev, 'dfdx': dfdx_prev, 'dhdx': dhdx_prev, 'dgdx': dgdx_prev})
                else:
                    # reset Hessian approximation to diagonal matrix (normal mode only)
                    H = self._reset_Hessian(mode='diag', H=H)
            else:
                # damped BFGS update
                s = x - x_prev
                dLdx_prev = dfdx_prev + np.dot(dhdx_prev.T, lam) + np.dot(dgdx_prev.T, sig)   # use the latest (not previous) Lagrange multipliers
                y = dLdx - dLdx_prev
                H = self._damped_BFGS(H, s, y)
            # END IF (Hessian approximation)

            # check if the Hessian is positive definite
            min_eval = np.min(np.linalg.eig(H)[0])
            if min_eval < -self.options['hessian_definite_tol']:   # relax a little bit
                print('   - WARNING: Hessian approximation is not positive definite! Minimum eval of Hessian =', min_eval)
                ### self.exit_status = 20
                ### break

                # reset Hessian to be positive definite
                if np.min(np.diag(H)) >= -self.options['hessian_definite_tol']:
                    # reset to diagonal matrix, which is positive definite
                    H = self._reset_Hessian(mode='diag', H=H)
                else:
                    # reset to scaled identity matrix, because the diagonal matrix is still not positive definite
                    H = self._reset_Hessian(mode='I_scaled', current_point={'x': x, 'dLdx': dLdx, 'lam': lam, 'sig': sig}, prev_point={'x': x_prev, 'dfdx': dfdx_prev, 'dhdx': dhdx_prev, 'dgdx': dgdx_prev})
            # END IF (Hessian check and reset)

            # --- solve QP and get search direction ---
            p_x, p_lam, p_sig, p_slack, success = self._solve_QP(H, dfdx, dhdx, h, dgdx, g, lam, sig, slack)

            if not success:
                # QP failed. Switch to elastic mode. If already in elastic mode, increase gamma and continue.
                print('   - QP solver failed. flag_elastic_mode =', self.elastic_mode)

                if not self.elastic_mode:
                    print('   - Entering elastic mode because QP solver failed.')
                    print('     Set elastic weight gamma = %1.1E' % self.gamma)
                    # setup elastic mode. update shape of x (now x <- [x, v, w, z]) and Lagrange multipliers for inequalities
                    self._setup_elastic_mode()
                else:
                    self.gamma *= 10.
                    print('   - WARNING: QP solver failed in elastic mode.')
                    print('     QP should not fail in elastic mode because in theory, elastic QP is always feasible. Something is wrong.')
                    print('     Increase gamma to %1.1E and continue.' % self.gamma)
                    # reset Hessian
                    H = self._reset_Hessian(mode='I_scaled', current_point={'x': x, 'dLdx': dLdx, 'lam': lam, 'sig': sig}, prev_point={'x': x_prev, 'dfdx': dfdx_prev, 'dhdx': dhdx_prev, 'dgdx': dgdx_prev})
                # END IF

                # solve QP (if needed, repeat by increasing gamma)
                gamma_max = 1e10
                while not success and self.gamma < gamma_max:
                    p_x, p_lam, p_sig, p_slack, success = self._solve_QP(H, dfdx, dhdx, h, dgdx, g, lam, sig, slack)

                    if not success:
                        self.gamma *= 10.
                        print('   - WARNING: QP solver failed in elastic mode.')
                        print('     Increase gamma to %1.1E and continue.' % self.gamma)
                # END WHILE
                if not success:
                    print('ERROR: QP solver failed in elastic mode, and gamma reached max(gamma). Exit.')
                    self.exit_status = 40
                    break
            # END IF (QP solver failed)

            # --- line search on [x, lam, sig] ---
            alpha, merit = self._line_search(x, p_x, lam, p_lam, sig, p_sig, slack, p_slack, H, backtrack=False)

            # if the line search fails, try a few recovery attempts
            if alpha is None:
                # TODO: reset Hessian to diagonal first?

                # 1. reset Hessian to scaled identity matrix
                if alpha is None:
                    print('   - WARNING: Line search failed.')
                    H = self._reset_Hessian(mode='I_scaled', current_point={'x': x, 'dLdx': dLdx, 'lam': lam, 'sig': sig}, prev_point={'x': x_prev, 'dfdx': dfdx_prev, 'dhdx': dhdx_prev, 'dgdx': dgdx_prev})
                    p_x, p_lam, p_sig, p_slack, success = self._solve_QP(H, dfdx, dhdx, h, dgdx, g, lam, sig, slack)
                    alpha, merit = self._line_search(x, p_x, lam, p_lam, sig, p_sig, slack, p_slack, H, backtrack=False)

                # 2. reset Hessian to identity if the line search still fails
                if alpha is None and not np.allclose(H, np.eye(self.nx)):
                    print('   - WARNING: Line search failed with scaled-identity Hessian.')
                    H = self._reset_Hessian(mode='I')
                    p_x, p_lam, p_sig, p_slack, success = self._solve_QP(H, dfdx, dhdx, h, dgdx, g, lam, sig, slack)
                    alpha, merit = self._line_search(x, p_x, lam, p_lam, sig, p_sig, slack, p_slack, H, backtrack=False)

                # 3. switch to elastic mode
                if alpha is None:
                    # failed line searches even after the Hessian reset
                    # Switch to elastic mode. If already in elastic mode, increase gamma and continue.
                    if not self.elastic_mode:
                        print('   - Entering elastic mode because line search failed.')
                        print('     Set elastic weight gamma = %1.1E' % self.gamma)
                        # setup elastic mode. update shape of x (now x <- [x, v, w, z]) and Lagrange multipliers for inequalities
                        self._setup_elastic_mode()
                    else:
                        self.gamma *= 10.
                        print('   - WARNING: Line search failed in elastic mode. Increase gamma to %1.1E and continue.' % self.gamma)
                    # END IF

                    # reset Hessian
                    H = self._reset_Hessian(mode='I_scaled', current_point={'x': x, 'dLdx': dLdx, 'lam': lam, 'sig': sig}, prev_point={'x': x_prev, 'dfdx': dfdx_prev, 'dhdx': dhdx_prev, 'dgdx': dgdx_prev})

                    # solve QP & perform line search (if needed, repeat by increasing gamma)
                    gamma_max = 1e10
                    while alpha is None and self.gamma < gamma_max:
                        p_x, p_lam, p_sig, p_slack, success = self._solve_QP(H, dfdx, dhdx, h, dgdx, g, lam, sig, slack,)

                        if success:
                            # perform line search
                            alpha, merit = self._line_search(x, p_x, lam, p_lam, sig, p_sig, slack, p_slack, H, backtrack=False)

                        if alpha is None:
                            print('   - WARNING: Either QP or line search failed in elastic mode.')
                            print('     QP success flag ='    , success, '| alpha =', alpha)
                            self.gamma *= 10.
                            print('     Increasing gamma to %1.1E and try again...' % self.gamma)
                        # END IF
                    # END WHILE
                    if alpha is None:
                        print('   - ERROR: Line search or QP solver failed in elastic mode, and gamma reached max(gamma). Exit.')
                        self.exit_status = 30
                        break
            # END IF (line search failed)

            # --- update variables ---
            # save the previous x and gradients (used for Hessian update)
            x_prev = np.copy(x)
            dfdx_prev = np.copy(dfdx)
            dhdx_prev = np.copy(dhdx)
            dgdx_prev = np.copy(dgdx)

            # update x and Lagrange multipliers
            x = x + alpha * p_x
            lam = lam + alpha * p_lam
            sig = sig + alpha * p_sig

            # evaluate functions and gradient/Jacobians at new x
            f = self._obj(x)
            dfdx = self._obj_grad(x)
            h, g = self._cons(x)
            dhdx, dgdx = self._cons_jac(x)
            dLdx = dfdx + np.dot(dhdx.T, lam) + np.dot(dgdx.T, sig)
            slack = self._compute_slack(g, sig)

            # optimality and feasibility at new x
            optimality, opt_l2 = self._compute_optimality(x, dfdx, dLdx, lam, sig)
            con_violation = np.concatenate((h, np.maximum(g, 0.)))
            feasibility = np.linalg.norm(con_violation, ord=np.inf) / max(1, np.linalg.norm(x, ord=np.inf))

            # Correct the initial/identity Hessian (Eq. 6.20 of Nocedal)
            if np.allclose(H, np.eye(self.nx)) and self.options['hessian_correction_for_identity']:
                s = x - x_prev
                dLdx_prev = dfdx_prev + np.dot(dhdx_prev.T, lam) + np.dot(dgdx_prev.T, sig)
                y = dLdx - dLdx_prev
                scaler = np.dot(s, y) / np.dot(y, y)

                if scaler <= 0.01 and scaler > 0:
                    # scaled identity istoo small, which wouldn't work well
                    scaler = 0.01
                elif scaler <= 0:
                    # negative curveature. Don't do scaled Hessian update
                    scaler = 1.
                # END IF

                print('   - Hessian corrected to scaled identity with scalar =', scaler)
                H = np.eye(self.nx) * scaler   # NOTE: this does not necessarily help
            
            # log major iteration history. discard elastic variables
            self.major_hist.append({'x': x[:self.nx], 'f': f, 'h': h, 'g': g[:self.ng], 'flag_elastic': self.elastic_mode, 'opt_inf': optimality, 'opt_l2': opt_l2, 'feas': feasibility})

            # increment iteration counter
            k += 1

            # --- print iteration info ---
            if self.elastic_mode:
                iter_info = 'i'
            else:
                iter_info = ' '

            if k % 20 == 0:
                print('Iter      Step     Feasible  Optimal   Merit         Penalty   ')
            print(' %3d %s  %1.1E  %1.2E  %1.2E  %+1.5E  %1.2E  %s' % (k, iter_print, alpha, feasibility, optimality, merit, np.linalg.norm(self.rho), iter_info))

            # --- exit elastic mode if the constraint violation is not too bad ---
            if self.elastic_mode:
                if feasibility <= 1e-2:
                    print('   - Exiting elastic mode because nonlinear infeasibilities are small enough.')
                    self.elastic_mode = False
                    # set back to original Hessian reset frequency
                    self.hessian_reset_freq = self.options['hessian_reset_freq']
                    # reset Hessian
                    H = self._reset_Hessian(mode='I')   # NOTE: resetting Hessian when exiting elastic mode improves the performance a lot! But maybe we should use the scaled Identity?
            # END IF (exit elastic mode)

        # END SQP main loop

        # check exit status
        if self.exit_status == -1:
            if k == self.options['max_iter']:
                self.exit_status = 10  # iteration limit reached
            else:
                self.exit_status = 0  # converged

        print('-------------------------------')
        # print('x_opt =', x)
        print('f_opt =', f)
        print('major iterations:', k, '| funcion calls:', self.n_calls_f, '| gradient calls:', self.n_calls_con)
        print('optimality =', optimality, '| feasibility =', feasibility)
        print('exit status:', self.exit_status)
        print('-------------------------------')

        # export major iterations history
        import pickle
        with open('major_hist.pkl', 'wb') as file:
            pickle.dump(self.major_hist, file)

        return x, f, self.exit_status

    def get_history(self):
        """ returns search history """
        return self.major_hist, self.f_hist
