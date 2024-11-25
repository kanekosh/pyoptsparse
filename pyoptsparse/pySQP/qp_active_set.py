"""
Active-set QP solver
Implements Algorithm 5.4 from Martins and Ning, Engineering Design Optimization
"""

import numpy as np
from scipy.linalg import null_space, cho_factor, cho_solve
import qpsolvers

def _check_linear_indep(A, Cw):
    """
    Check if the rows of the matrix [A; Cw] are linearly independent
    """
    if A.shape[0] == 0:
        mat = Cw
    else:
        # concatenate A and Cw to get the full contstraint matrix
        ### mat = np.vstack((A, Cw))
        mat = Cw   # HACK!! ignore equality constraints

    if mat.shape[0] == 0:
        return True

    # see if A is full row rank
    if np.linalg.matrix_rank(mat) == mat.shape[0]:
        return True
    else:
        return False


def _find_feasible_point(A, b, C, d, x0):
    """
    Find a closest feasible point to x0 that satisfies the constraints:
    A * x + b = 0
    C * x + d <= 0

    Minimize 1/2 ||x - x0||^2 subject to the constraints.
    -> equivalent objective is: 1/2 x' * I * x - x0' * x
    """

    if A.shape[0] == 0:
        # no equality constraints
        qp_prob = qpsolvers.Problem(np.eye(len(x0)), -x0, G=C, h=-d)
    else:
        qp_prob = qpsolvers.Problem(np.eye(len(x0)), -x0, G=C, h=-d, A=A, b=-b)

    qp_sol = qpsolvers.solve_problem(qp_prob, solver='gurobi', initvals=x0)

    if not qp_sol.found:
        raise ValueError('Failed to find a feasible point in QP phase-1 problem')

    return qp_sol.x


def qp(Q, q, A, b, C, d, x0, W=None):
    """
    Solve the following QP:
    min   0.5 * x' * Q * x + q' * x
    s.t.  A * x + b = 0
          C * x + d <= 0

    Parameters
    ----------
    Q : ndarray
        positive semi-definite Hessian matrix
    q : ndarray
        vector
    A : ndarray
        matrix (equality constraint Jacobian)
    b : ndarray
        vector (equality constraint RHS)
    C : ndarray
        matrix (inequality constraint Jacobian)
    d : ndarray
        vector (inequality constraint RHS)
    x0 : ndarray
        initial guess.
        This must be a feasible point
    W : list of int
        initial guess of working set

    Returns
    -------
    x : ndarray
        optimal x for QP
    lam : ndarray
        Lagrange multipliers for equality constraints
    sig : ndarray
        Lagrange multipliers for inequality constraints
    """

    k_max = 1000   # max QP iterations
    tol = 1e-10

    # problem dimensions
    nx = len(x0)   # number of variables
    n_eq = len(b)  # number of equality constraints
    n_ineq = len(d)  # number of inequality constraints

    # modify shape of the constraint matrices
    if n_eq == 0:
        A = np.empty((0, nx))
    if n_ineq == 0:
        C = np.empty((0, nx))

    # --- sanity check: Q matrix must be positive semi-definite ---
    min_eval = np.min(np.linalg.eig(Q)[0])
    if min_eval <= -1e-10:   # relax a little bit
        raise ValueError('QP ERROR: Q matrix is not positive (semi-)definite. Min eval=', min_eval)

    # check if Q is strictly positive definite
    if min_eval < 1e-14:
        print('WARNING: Q matrix is not strictly positive definite. This may cause numerical issues.')
        flag_semidef = True
    else:
        flag_semidef = False

    # --- check if initial point is feasible ---
    if n_eq > 0:
        cons_eq = np.dot(A, x0) + b
    else:
        cons_eq = 0.
    if n_ineq > 0:
        cons_ineq = np.dot(C, x0) + d
    else:
        cons_ineq = 0.

    if np.any(abs(cons_eq) > tol) or np.any(cons_ineq > tol):
        # initial point is not feasible, so solve a phase-1 problem to find a feasible point
        xk = _find_feasible_point(A, b, C, d, x0)
        print('Given initial point is not feasible for QP. Solved the phase-1 QP to find a closest feasible point.')
        ### print(f'New initial x = {xk}')
    else:
        # initial point is feasible. Continue
        xk = x0.copy()

    # --- initial guess of the working set (if not provided) ---
    if W is None:
        active_cons = np.dot(C, xk) + d > -1e-10
        W = list(np.arange(n_ineq, dtype=int)[active_cons])
        # check linear dependency of the working set constraints
        if not _check_linear_indep(A, C[W, :]):
            # current working set includes linearly dependent constraints, so we must remove some
            remove_cons_count = 0
            while not _check_linear_indep(A, C[W, :]):
                W.pop()
                remove_cons_count += 1
            print(f'Removed {remove_cons_count} constraints from the initial working set to make it linear independent')

    # TODO: the reduced Hessian for the initial working set should be positive definite - may need to remove bound constraints from linear variables to achieve this!
    #       Or we might need "temporary" constraints to achieve this. See page 6 of Gill 1991 (Inertia-controlling methods)
    # --- QP main loop ---
    
    for k in range(k_max):
        # --- solve KKT system for the current working set ---
        # Eq. (5.81) from the MDO book (Martins and Ning)
        # NOTE: This approach fails when Q matrix is singular (this happens in the elastic mode where Q is semi-definite due to linear elastic variables)
        #       One workaround would be regularizing Q matrix by adding small value to the diagonal to make it positive definite.
        #       Also, this approach is not very efficient because np.linalg.solve does not exploit the symmetric structure of the KKT matrix.
        #       More advanced methods like null-space method can be more efficient?
        kkt_dim = nx + n_eq + len(W)

        mat = np.zeros((kkt_dim, kkt_dim))
        mat[:nx, :nx] = Q
        if n_eq > 0:
            mat[:nx, nx:nx + n_eq] = A.T
            mat[nx:nx + n_eq, :nx] = A
        if len(W) > 0:
            mat[:nx, nx + n_eq:] = C[W, :].T
            mat[nx + n_eq:, :nx] = C[W, :]

        rhs = np.zeros(kkt_dim)
        rhs[:nx] = -q - np.dot(Q.T, xk)

        if not flag_semidef:
            sol = np.linalg.solve(mat, rhs)
        else:
            # Q matrix is semi-definite, hense the KKT matrix is singular. We cannot solve the KKT system directly.
            # Find a search direction by solving mat @ [p; lambda; sigma] = 0 for non-zero solution.
            raise NotImplementedError('Q matrix is semi-definite, so we need some sort of modification to solve the KKT system. This is not implemented yet.')

        p = sol[:nx]
        lam = sol[nx:nx + n_eq]   # multipliers for equality constraints
        sig = np.zeros(n_ineq)
        sig[W] = sol[nx + n_eq:]   # multipliers for inequality constraints. Non-active constraints will have sig=0

        # --- null-space method ---
        # # TODO: null-space is empty when con_mat is shape (n, n). In that case I can just solve con_mat @ (x + p) = b directly?
        # rhs = q + np.dot(Q.T, xk)
        # con_mat = np.vstack((A, C[W, :]))
        # Z = null_space(con_mat)
        # # solve reduced-Hessian linear system by Cholesky factorization
        # reduced_Hessian = np.dot(Z.T, np.dot(Q, Z))
        # reduced_rhs = -np.dot(Z.T, rhs)
        # pz = cho_solve(cho_factor(reduced_Hessian), reduced_rhs)   # NOTE: this is `dz` in Gill 2005
        # p_NULL = np.dot(Z, pz)  # step for x
        # # compute Lagrange multipliers by solving Eq. (16.20) from Nocedal
        # Y = null_space(Z.T)  # given Z of shape (n, n-m), Y is of shape (n, m) such that [Y|Z] is non-singular. See Nocedal Ch 16.2
        # mat_for_lambda = np.dot(con_mat, Y).T
        # rhs_for_lambda = -np.dot(Y.T, rhs + np.dot(Q, p))
        # lagrange_multipliers = np.linalg.solve(mat_for_lambda, rhs_for_lambda)
        
        # lam_NULL = lagrange_multipliers[:n_eq]
        # sig_NULL = np.zeros(n_ineq)
        # sig_NULL[W] = lagrange_multipliers[n_eq:]   # multipliers for inequality constraints. Non-active constraints will have sig=0

        # print('\n\n\n---')
        # --- end of null-space method ---

        if np.linalg.norm(p) < tol:
            if np.all(sig >= 0):
                # solution satisfied the KKT condition; exit
                print(f'QP (active-set) Converged in {k} iterations')
                # print(f'x = {xk}')
                # print(f'lambda = {lam}')
                # print(f'sigma = {sig}')
                return xk, lam, sig, W
            else:
                # remove a constraint from the working set
                W.remove(np.argmin(sig))
        
        else:   # (p != 0)
            # determine the step size such that non-working set constraints are satisfied
            # NOTE: when the reduced Hessian is indefinite, alpha can be greater than 1.0
            non_W = list(set(range(n_ineq)) - set(W))   # list of non-working set indices
            C_non_W = C[non_W, :]

            # find the blocking constraint and corresponding step size
            alpha_all = -(np.dot(C_non_W, xk) + d[non_W]) / np.dot(C_non_W, p)
            alpha_all[np.dot(C_non_W, p) <= 0] = 1.0   # set alpha = 1 for constraints that are satisfied, because they are non-blocking
            min_alpha = np.min(alpha_all)
            if min_alpha < 1.0:
                # add a blocking constraint with min alpha to the working set
                blocking_idx = non_W[np.argmin(alpha_all)]
                W.append(blocking_idx)
            
                # check linear independence of the working set constraints
                if not _check_linear_indep(A, C[W, :]):
                    # the blocking constraint we just added is linearly dependent with the existing working-set constraints, so remove it
                    W.remove(blocking_idx)
                    print('Not adding a blocking constraint because it is linearly dependent')

            # update xk and continue
            xk = xk + min_alpha * p
        # END IF

        # print(f'Iteration {k}: x = {xk}, W = {W}')
        print(f'Iteration {k}')
    # END FOR

    raise ValueError('Maximum number of iterations reached')


if __name__ == '__main__':
    # --- test problem from MDO book Example 5.10 ---
    # Q = np.array([[6, 2], [2, 2]])
    # q = np.array([1, 6])
    # C = np.array([[-2, -3], [-1, 0], [0, -1]])
    # d = np.array([4, 0, 0])

    # x0 = np.array([-2., 2.])

    # qp(Q, q, np.array([]), np.array([]), C, d, x0)

    # --- test problem from Nocedal Example 16.4 ---
    Q = np.array([[2, 0], [0, 2]])
    q = np.array([-2, -5])
    C = -np.array([[1, -1, -1, 1, 0], [-2, -2, 2, 0, 1]]).T
    d = -np.array([2, 6, 2, 0, 0])

    x0 = np.array([100., 100.])
    
    qp(Q, q, np.array([]), np.array([]), C, d, x0)