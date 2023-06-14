"""
Wrapper of my SQP
"""

# Standard Python modules
import copy
import datetime
import time

# External modules
import numpy as np

# Local modules
from ..pyOpt_error import Error
from ..pyOpt_optimizer import Optimizer
from ..pyOpt_utils import ICOL, INFINITY, IROW, convertToCOO, extractRows, scaleRows

# import SQP optimizer
import os
import sys
sys.path.append('/Users/shugo/rsrc/SQP/')
from sqp import SQP as SQPmain


class SQP(Optimizer):
    """
    SQP optimizer class
    """

    def __init__(self, raiseError=True, options={}):
        name = "SQP"
        category = "Local Optimizer"
        defOpts = self._getDefaultOptions()
        informs = self._getInforms()

        super().__init__(
            name,
            category,
            defaultOptions=defOpts,
            informs=informs,
            options=options,
            checkDefaultOptions=False,
        )

        self.jacType = "dense2d"

    @staticmethod
    def _getInforms():
        informs = {
            -1: "Setup failed",
            0: "Succedded",
            10: "Iteration limit exceeded",
            20: "Hessian approx became not positive semi-definite",
            30: "Line search failed",
        }
        return informs

    @staticmethod
    def _getDefaultOptions():
        defOpts = {}
        return defOpts

    def __call__(
        self,
        optProb,
        sens=None,
        sensStep=None,
        sensMode=None,
        storeHistory=None,
        hotStart=None,
        storeSens=True,
    ):
        """
        This is the main routine used to solve the optimization
        problem.

        Parameters
        ----------
        optProb : Optimization or Solution class instance
            This is the complete description of the optimization problem
            to be solved by the optimizer

        sens : str or python Function.
            Specifiy method to compute sensitivities.  To explictly
            use pyOptSparse gradient class to do the derivatives with
            finite differenes use \'FD\'. \'sens\' may also be \'CS\'
            which will cause pyOptSpare to compute the derivatives
            using the complex step method. Finally, \'sens\' may be a
            python function handle which is expected to compute the
            sensitivities directly. For expensive function evaluations
            and/or problems with large numbers of design variables
            this is the preferred method.

        sensStep : float
            Set the step size to use for design variables. Defaults to
            1e-6 when sens is \'FD\' and 1e-40j when sens is \'CS\'.

        sensMode : str
            Use \'pgc\' for parallel gradient computations. Only
            available with mpi4py and each objective evaluation is
            otherwise serial

        storeHistory : str
            File name of the history file into which the history of
            this optimization will be stored

        hotStart : str
            File name of the history file to "replay" for the
            optimziation.  The optimization problem used to generate
            the history file specified in \'hotStart\' must be
            **IDENTICAL** to the currently supplied \'optProb\'. By
            identical we mean, **EVERY SINGLE PARAMETER MUST BE
            IDENTICAL**. As soon as he requested evaluation point does
            not match the history, function and gradient evaluations
            revert back to normal evaluations.

        storeSens : bool
            Flag sepcifying if sensitivities are to be stored in hist.
            This is necessay for hot-starting only.
        """
        self.startTime = time.time()
        self.callCounter = 0
        self.storeSens = storeSens

        self.userRequestedTermination = False

        if len(optProb.constraints) == 0:
            self.unconstrained = True
            # TODO: activate dummy constraint?
            # optProb.dummyConstraint = True
            raise NotImplementedError("SQP currently only supports constrained problems")

        # Save the optimization problem and finalize constraint
        self.optProb = optProb
        self.optProb.finalize()
        # Set history/hotstart
        self._setHistory(storeHistory, hotStart)
        self._setInitialCacheValues()
        blx, bux, xs = self._assembleContinuousVariables()
        self._setSens(sens, sensStep, sensMode)

        # TODO: currently, the variables bound blx, bux are ignored
        if np.min(np.abs(blx)) < 1e10 or np.min(np.abs(bux)) < 1e10:
            print("\n*** Warning: SQP currently ignores variable bounds ***\n")

        # setup constraint Jacobian
        indices, blc, buc, fact = self.optProb.getOrdering(["ne", "le", "ni", "li"], oneSided=True)
        self.optProb.jacIndices = indices
        self.optProb.fact = fact
        self.optProb.offset = buc

        # ---!!!--- SQP currently only supports equality ---!!!---
        if not np.allclose(blc, buc, 1e-10, 1e-10):
            raise NotImplementedError("My SQP does not support inequality constraints")

        # We make a split here: If the rank is zero we setup the
        # problem and run SQP, otherwise we go to the waiting loop:

        if self.optProb.comm.rank == 0:

            # Define call back functions to be supplied to SQP
            def eval_obj(x):
                """ objective """
                fobj, fail = self._masterFunc(x, ["fobj"])
                if fail == 2:
                    self.userRequestedTermination = True
                return fobj.copy()

            def eval_obj_grad(x):
                """ objective gradient """
                gobj, fail = self._masterFunc(x, ["gobj"])
                if fail == 2:
                    self.userRequestedTermination = True
                return gobj.copy()

            def eval_cons(x):
                """ constraints h=0 """
                fcon, fail = self._masterFunc(x, ["fcon"])
                if fail == 2:
                    self.userRequestedTermination = True
                return fcon.copy()

            def eval_cons_jac(x):
                """ constraint Jacobian """
                gcon, fail = self._masterFunc(x, ["gcon"])
                if fail == 2:
                    self.userRequestedTermination = True
                return gcon.copy()

            # TODO: add callback to SQP for user termination
            # Define intermediate callback. If this method returns false,
            # Ipopt will terminate with the User_Requested_Stop status.
            # def eval_intermediate_callback(*args, **kwargs):
            #     if self.userRequestedTermination is True:
            #         return False
            #     else:
            #         return True

            timeA = time.time()

            # setup and run SQP
            sqp = SQPmain(nx=self.optProb.ndvs, nh=self.optProb.nCon, obj=eval_obj, grad_obj=eval_obj_grad, cons=eval_cons, grad_cons=eval_cons_jac)
            x_opt, obj_opt, status = sqp.optimize(xs)

            optTime = time.time() - timeA

            if self.storeHistory:
                self.metadata["endTime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.metadata["optTime"] = optTime
                self.hist.writeData("metadata", self.metadata)
                self.hist.close()

            # Store Results
            sol_inform = {}
            sol_inform["value"] = status
            sol_inform["text"] = self.informs[status]

            # Create the optimization solution
            sol = self._createSolution(optTime, sol_inform, obj_opt, x_opt)

            # Indicate solution finished
            self.optProb.comm.bcast(-1, root=0)
        else:
            self._waitLoop()
            sol = None

        # Communication solution and return
        sol = self._communicateSolution(sol)

        return sol
