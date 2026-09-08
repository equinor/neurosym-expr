import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
from jax import grad, hessian, jit
from jax.scipy.linalg import block_diag
import copy
import jax.debug

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

class adaptive_spline_transport():
    
    def __init__(self, k = None, degree = 3, skip_dimensions = 0):
        
        # Create the jitted functions
        self.create_jitted_functions()
        
        # Set the number of knots and degree of the spline
        self.k = k
        self.degree = degree
        
        # Do we need skip dimensions?
        self.skip_dimensions = skip_dimensions
        
        # Set the StandardScaler
        self.scaler = None
        
        return
    
    def train_map(self, X, sparsity = None, lambda_initial = 2.0, optimize_lambdas = True, beta_initial = None):
        
        """
        This function takes as input an N-by-D array of training samples X, 
        then creates a triangular map based on P-splines according to the 
        class object's specified properties.
        
        For the time being, I will assume the samples have been standardized.
        """
        
        # =====================================================================
        # PREPARE THE MAP TRAINING
        # =====================================================================
        
        # Standardize the samples
        self.scaler = StandardScaler()
        
        # Scale the samples
        X = self.scaler.fit_transform(copy.copy(X))
        
        # Create a copy of the input
        self.X = copy.copy(X)
        
        # Learn the number of samples and dimensions
        self.N,self.D = X.shape
        
        # Create the sparsity matrix
        if sparsity is not None:
            assert sparsity.shape[0] + self.skip_dimensions == sparsity.shape[1], "sparsity matrix must be square. Together wth skip_dimensions, it currently has shape ({} + {}, {}).".format(sparsity.shape[0],self.skip_dimensions,sparsity.shape[1])
            assert np.all((sparsity == 0) | (sparsity == 1)), "sparsity matrix must only contain values of 0 (no dependence) and 1 (dependence)."
            self.sparsity = sparsity
        else:
            self.sparsity = np.tri(self.D)[self.skip_dimensions:,:]
    
        # Derive the number of knots from the sample size, if it hasn't been assigned
        if self.k is None:
            self.k = int(np.ceil(self.N**(1./3)))
            
        # Create the basis spline functions for each marginal
        self.Bx = {} # Basis spline functions
        self.X_basis = {} # Basis spline function evaluations
        self.nbases = {} # Number of basis functions in each term
        self.S_list = {} # List of smoothing matrices for the basis spline functions
        self.lambdas = {} # Optimal smoothing coefficients of the map component functions
        self.betas = {} # Optimal spline coefficients of the map component functions
        self.training_data_range = {} # Marginal range of the training data, used to extrapolate root finding
        self.dofs = np.zeros(self.D)
        self.aics = np.zeros(self.D)
        self.nlls = np.zeros(self.D)
        
        # =====================================================================
        # Pre-compute all the evaluations for the skipped dimensions
        # =====================================================================
        
        for d in range(self.skip_dimensions):
            
            # Save the training data range
            self.training_data_range[d] = [np.min(self.X[:,d]),np.max(self.X[:,d])]
            
            # Create the basis spline function for this marginal
            self.Bx[d] = self.create_bspline_basis(
                self.X[:,d], 
                k = self.k, 
                degree = self.degree)
            
            # Evaluate the basis spline function for this marginal
            self.X_basis[d] = self.Bx[d](self.X[:,d])
            
        # =====================================================================
        # Go through each dimension and build the corresponding map component function
        # =====================================================================
        
        for d in np.arange(self.skip_dimensions,self.D,1):
            
            print("Now training map component {} (number dependencies {})".format(d,np.sum(self.sparsity[d-self.skip_dimensions,:])),end="\n")
            
            # Save the training data range
            self.training_data_range[d] = [np.min(self.X[:,d]),np.max(self.X[:,d])]
            
            # Create the basis spline function for this marginal
            self.Bx[d] = self.create_bspline_basis(
                self.X[:,d], 
                k = self.k, 
                degree = self.degree)
            
            # Evaluate the basis spline function for this marginal
            self.X_basis[d] = self.Bx[d](self.X[:,d])
            
            # Depending on sparsity, create the following for this map component function:
            # - a dictionary of smoothing matrices
            # - a dictionary of the number of basis functions
            # - a dictionary of basis function evaluations
            S_list_local = {}
            nbasis_local = {}
            X_basis_local = {}
            for j in range(d): # Go through every nonmonotone dependence
                if self.sparsity[d-self.skip_dimensions,j] == 1: # If this dependency exists in the sparsity matrix
                    nbasis_local[j] = self.X_basis[j].shape[1]
                    S_list_local[j] = self.create_smoothing_matrix(nbasis_local[j])
                    X_basis_local[j] = self.X_basis[j]
            nbasis_local[d] = self.X_basis[d].shape[1]
            S_list_local[d] = self.create_smoothing_matrix(nbasis_local[d])
            X_basis_local[d] = self.X_basis[d]
                
            # Append these lists to the collection
            self.nbases[d] = copy.deepcopy(nbasis_local)
            self.S_list[d] = copy.deepcopy(S_list_local)
            
            # Get the partial derivative of the spline basis functions wrt the last argument
            dBxj = self.Bx[d].derivative()(self.X[:,d])
            
            # Calculate the initial lambda
            if type(lambda_initial) != dict:
                # one lambda per active block (all non-mono parents + monotone)
                lambda_initial_d = [lambda_initial] * len(X_basis_local)
            else:
                lambda_initial_d = copy.copy(lambda_initial[d])
            
            
            # =================================================================
            # Solve the outer optimization problem, if required
            # =================================================================
            
            if optimize_lambdas: # Yes, we do solve the outer optimization problem
            
                # Solve the outer optimization problem
                lambda_opt = minimize(
                    self.outer_optimization_obj_and_jac, 
                    lambda_initial_d,
                    args = (
                        [X_basis_local[key] for key in list(X_basis_local.keys())],
                        self.S_list[d],
                        dBxj,
                        d
                    ),
                    bounds = [(-10,10) for entry in lambda_initial_d],
                    method = "L-BFGS-B",
                    jac = True
                )
                
                # Extract the optimal lambda value
                self.lambdas[d] = lambda_opt.x # np.array([-1e1])
                
            else: # No, we do not solve the outer optimization problem
                
                # Continue with the initial lambda values
                self.lambdas[d] = np.asarray(lambda_initial_d)
            
            # =================================================================
            # Solve the inner optimization problem to get the optimal coefficients
            # =================================================================
            
            beta = self.find_optimal_coefficients(
                log_lambda  = self.lambdas[d], 
                X_basis     = [X_basis_local[key] for key in list(X_basis_local.keys())], 
                S_list      = self.S_list[d], 
                dBxj        = dBxj, 
                k           = d, 
                beta_initial = beta_initial)
            
            # Append the optimal betas
            self.betas[d] = copy.copy(beta)
            
            self.dofs[d] = self.effective_degrees_of_freedom(
                beta_opt    = self.betas[d], 
                log_lambda  = self.lambdas[d], 
                X_basis     = [X_basis_local[key] for key in list(X_basis_local.keys())], 
                S_list      = self.S_list[d], 
                dBxj        = dBxj)
            self.aics[d] = self.AIC(
                beta_opt    = self.betas[d], 
                log_lambda  = self.lambdas[d], 
                X_basis     = [X_basis_local[key] for key in list(X_basis_local.keys())], 
                S_list      = self.S_list[d], 
                dBxj        = dBxj)
            self.nlls[d] = self.nll_jit(
                beta_init   = self.betas[d], 
                X_basis     = [X_basis_local[key] for key in list(X_basis_local.keys())], 
                dBxj        = dBxj)
            
    def apply_forward_map(self, X = None):
        
        # Create an empty matrix of pushfoward samples
        Z   = np.zeros((X.shape[0],X.shape[1] - self.skip_dimensions))
         
        # If X has been provided, re-compute the bases
        if X is not None: 
            
            # Transform the samples
            X   = self.scaler.transform(copy.copy(X))
            
            X_basis_local = {}
            
            # Go through each dimension and build the corresponding map component function
            for d in np.arange(0,self.D,1):
                
                # Evaluate the basis spline function
                X_basis_local[d] = self.Bx[d](X[:,d])
        
        # If X has not been provided, read the basis function evaluations
        else:
            
            # Copy from memory
            X_basis_local = copy.deepcopy(self.X_basis)
        
        
        # Go through each dimension and apply the pushforward map
        for d in np.arange(self.skip_dimensions,self.D,1):
            
            # Reparameterize the betas
            # For the monotone component, the coefficients represent an offset
            # and then beta increments. This function converts them back into
            # ascending coefficients
            
            beta_all = np.asarray(self.betas[d])
            nb = self.nbases[d][d]
            beta_mon = self.reparametrize(beta_all[-nb:])
            beta_reparam = np.concatenate([beta_all[:-nb], beta_mon])
            
            # Find all basis function evaluations, given sparsity
            basis_evals = []
            for j in range(d+1):
                if self.sparsity[d - self.skip_dimensions,j] == 1:
                    basis_evals.append(X_basis_local[j])
            
            Z[:,d - self.skip_dimensions]  = np.hstack(basis_evals).dot(beta_reparam)
            
        return Z
    
    def apply_inverse_map(self, Z):
        
        # Create a local copy of Z
        Z   = copy.copy(Z)
        
        # Create an empty matrix of pullback samples
        X   = np.zeros(Z.shape)
        
        # Go through each dimension and apply the pushforward map
        for d in np.arange(self.skip_dimensions,self.D,1):
            
            # Invert the map component function
            X      = self.vectorized_root_search_alternate(
                Zk          = Z[:,d - self.skip_dimensions],
                X           = X,
                d           = d)
            
        # Undo the standardization
        X   = self.scaler.inverse_transform(X)
        
        return X
    
    def apply_conditional_inverse_map(self, X_star, Z):
        
        assert X_star.shape[0] == Z.shape[0], "X_star (N="+str(X_star.shape[0])+") and Z (N="+str(Z.shape[0])+") must have the same number of samples."
        assert X_star.shape[1] + Z.shape[1] == self.D, "Dimensions of X_star ("+str(X_star.shape[1])+") and Z ("+str(Z.shape[1])+") must add up to "+str(self.D)+"."
        assert X_star.shape[1] == self.skip_dimensions, "The dimensions of X_star should equal skip_dimensions, but they are {} and {}, respectively.".format(X_star.shape[1], self.skip_dimensions)
        
        # Create a copy of X_star
        X_star = copy.copy(X_star)
        
        # Standardize X_star
        X_star -= self.scaler.mean_[:X_star.shape[1]]
        X_star /= self.scaler.scale_[:X_star.shape[1]]
        
        # Create an empty matrix of pullback samples
        X   = np.hstack([X_star,copy.copy(Z)])
        
        # Go through each dimension and apply the pushforward map
        for d in np.arange(X_star.shape[1],self.D,1):
            
            # Invert the map component function
            X      = self.vectorized_root_search_alternate(
                Zk          = Z[:,d - X_star.shape[1]],
                X           = X,
                d           = d)
            
        # Undo the standardization
        X   = self.scaler.inverse_transform(X)
            
        return X
    
    #%%
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    # Re-parametrize coefficients to ensure monotonicity
    def reparametrize(self, beta_increments):
        
        #return beta_increments
        increments_transformed = jnp.array([beta_increments[0]] + list(jax.nn.softplus(beta_increments)[1:]))
        return jnp.cumsum(increments_transformed)

    # Define B-spline basis functions with knots at quantiles
    def create_bspline_basis(self, x, k=10, degree=3):

        knots = np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), k + 2)  # Define quantiles
        knots = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))  # Extend knots for spline degree
        bspline = BSpline(knots, np.eye(len(knots) - degree - 1), degree)

        return self.BSplineLinearized(bspline, degree, knots)

    class BSplineLinearized:
        def __init__(self, bspline, degree, knots):
            self.bspline = bspline
            self.degree = degree
            self.knots = knots
            
            # Find the "first" and "last" real knot
            self.first_knot = self.knots[self.degree] # The (self.degree - 1) offsets the knots so that the second derivative will be zero
            self.last_knot = self.knots[-(self.degree + 1)] # The (self.degree - 1) offsets the knots so that the second derivative will be zero
            
            # Compute the number of coefficients
            self.num_coeffs = len(self.knots) - self.degree - 1
            
            # Get the tail derivatives
            self.spline_derivative = self.bspline.derivative(nu = 1) # First order derivative
            
            # Compute the offset and the slope at the training ensemble's edges
            self.offset_low = self.bspline(self.first_knot)
            self.slope_low = self.spline_derivative(self.first_knot)
            self.offset_high = self.bspline(self.last_knot)
            self.slope_high = self.spline_derivative(self.last_knot)
    
        def __call__(self, x):
            
            # Pre-allocate space for the basis function evaluations
            basis = np.zeros((len(x),self.num_coeffs))
            
            # Find which samples are below, within, and above the training ensemble
            below = np.where(x < self.first_knot)[0]
            within = np.where(np.logical_and(x >= self.first_knot, x <= self.last_knot))[0]
            above = np.where(x > self.last_knot)[0]
            
            # Evaluate the spline below the training ensemble
            if len(below) >= 1:
                basis[below,:] = self.offset_low[None,:] + (np.repeat(x[below][:,None],repeats=self.num_coeffs,axis=-1) - self.first_knot)*self.slope_low[None,:]
                
            # Evaluate the spline within the training ensemble
            if len(within) >= 1:
                basis[within,:] = self.bspline(x[within])

            # Evaluate the spline above the training ensemble
            if len(above) >= 1:
                basis[above,:] = self.offset_high[None,:] + (np.repeat(x[above][:,None],repeats=self.num_coeffs,axis=-1) - self.last_knot)*self.slope_high[None,:]
    
            return basis
    
        def derivative(self):
            
            # needed to mock old BSpline api
            def mock_function(x):
                
                # Crop samples outside the training ensemble to the edges
                x = np.minimum(np.maximum(self.first_knot,copy.copy(x)),self.last_knot)
                
                # Evaluate the derivatives and return them
                return self.spline_derivative(x)
            
            return mock_function
        
        def second_derivative(self):
            
            # needed to mock old BSpline api
            def mock_function(x):
                
                # Crop samples outside the training ensemble to the edges
                x = np.minimum(np.maximum(self.first_knot,copy.copy(x)),self.last_knot)
                
                # Evaluate the derivatives and return them
                return self.bspline.derivative(nu = 2)(x)
            
            return mock_function
        
        
    # Define S_block_lambda using JAX-compatible operations
    def S_block_lambda(self, S_list, lambdas_smooth):
        assert len(lambdas_smooth) == len(S_list), "S_list and lambdas_smooth must be of same length"
        blocks = [lambdas_smooth[idx] * S_list[key] for idx,key in enumerate(list(S_list.keys()))]
        return block_diag(*blocks)

    def create_constrained_smoothing_matrix(self, nbasis):
        P = np.zeros((nbasis-2, nbasis))
        for i in range(nbasis-2):
            P[i,i+1] = 1
            P[i,i+2] = -1
        return P.T @ P
    
    # Define the smoothing matrix S
    def create_smoothing_matrix(self, nbasis):
        """For p-splines https://stats.stackexchange.com/questions/512045/generalized-additive-models-what-exactly-is-being-penalized-when-using-a-p-spli"""
        D = np.diff(np.eye(nbasis), n=2) # n=2: the difference degree
        S = D @ D.T
        return S
    
    def monotone_optimization_objective(self, beta_init, X_basis, dBxj, log_lambda, S_list):
        
        # Get the vector of monotone coefficients
        beta_mon = self.reparametrize(beta_init)
        
        # Convert log λ → λ_eff = exp(log λ) * s_j^2  (block-wise scaling)
        lam = jnp.exp(log_lambda)
        scale_vec = jnp.asarray([jnp.sqrt(jnp.trace(P.T @ P) / P.shape[0]) for P in X_basis])
        lam_eff = lam * (scale_vec ** 2)
        
        # Apply scaling to each block's S
        S_list = [S_list[key] * lam_eff[idx] for idx, key in enumerate(list(S_list.keys()))]
        
        # =====================================================================
        
        # We have a slightly simpler optimization if there are no nonmonotone terms
        if len(X_basis) == 1: # There are no nonmonotone terms
        
            # Prepare the A array
            P_mon = X_basis[-1]
            
            # Create the block matrix for the smoothing penalty
            S_mon = S_list[-1]
            
            # Compute the objective
            objective = 0
            
            # Quadratic term
            objective = jnp.sum((P_mon @ beta_mon)**2)/2
            
            # Subtract the log barrier
            logarg = dBxj @ beta_mon
            objective -= jnp.sum(jnp.log(logarg))
            
            # Add the smoothing penalty 
            objective += 1/2 * beta_mon.T @ S_mon @ beta_mon
        
        else: # There are nonmonotone terms
            
            # Prepare the A array
            P_nonmon = jnp.hstack(X_basis[:-1])
            P_mon = X_basis[-1]
            
            # Create the block matrix for the smoothing penalty
            S_nonmon = block_diag(*S_list[:-1])
            S_mon = S_list[-1]
            
            # Compute Mk with ridge and solve (more stable than explicit inverse)
            A = P_nonmon.T @ P_nonmon + S_nonmon
            
            # ridge scaled to problem size for numerical stability
            rid = 1e-6 * jnp.maximum(1.0, jnp.mean(jnp.diag(A)))
            A = A + rid * jnp.eye(A.shape[0])
            Mk = jnp.linalg.solve(A, P_nonmon.T)
           
            # Compute Ak and Dk
            Ak = (np.identity(P_mon.shape[0]) - P_nonmon @ Mk) @ P_mon
            Dk = Mk @ P_mon
            
            # Compute the objective
            objective = 0
            
            # Quadratic term
            objective = jnp.sum((Ak @ beta_mon)**2)/2
            
            # Subtract the log barrier
            logarg = dBxj @ beta_mon
            objective -= jnp.sum(jnp.log(logarg))
            
            # Add the smoothing penalty 
            objective += 1/2 * beta_mon.T @ (Dk.T @ S_nonmon @ Dk + S_mon) @ beta_mon
            
        # jax.debug.print("objective {obj:.6f}", obj=objective)
        
        return objective

    def nll(self, beta_init, X_basis, dBxj):
        
        """
        This function evaluates the negative log-likelihood.
        """
        
        # Concatenate into a single matrix of basis function evaluations
        X_full = jnp.hstack(X_basis)
            
        # How many basis function terms are in the monotone part
        nbasis_mono = X_basis[-1].shape[1]
        
        # Reparameterize the beta coefficients
        beta_reparam = beta_init.copy()
        beta_reparam = beta_reparam.at[-nbasis_mono:].set(self.reparametrize(beta_reparam[-nbasis_mono:]))
        
        # Matrix multiplication of basis function evaluations and reparameterized beta coefficients
        Sx = X_full @ beta_reparam

        # Do the same for the partial derivatives w.r.t. the monotone coefficients
        dSxj = dBxj @ beta_reparam[-nbasis_mono:]
        
        # Return the objective
        return 0.5 * jnp.sum(Sx**2) - jnp.sum(jnp.log(dSxj))

    def nll_penalized(self, beta_init, log_lambda, X_basis, S_list, dBxj):
        # lambdas
        lam = jnp.atleast_1d(jnp.exp(log_lambda))
        # Block-wise scaling (A): per-block basis scale s_j = sqrt(tr(P_j^T P_j)/N)
        scale_vec = jnp.asarray([jnp.sqrt(jnp.trace(P.T @ P) / P.shape[0]) for P in X_basis])
        lam_eff = lam * (scale_vec ** 2)
    
        # split into nonmonotone increments and monotone increments
        nb_mono = X_basis[-1].shape[1]                      # size of monotone block
        beta_inc = beta_init[-nb_mono:]                     # increments (mono)
        beta_non = beta_init[:-nb_mono]                     # non-mono coeffs (maybe empty)
    
        # reparametrize the monotone block: c = cumsum([β0, softplus(β1:)] )
        c = self.reparametrize(beta_inc)                    # matches inner objective, :contentReference[oaicite:0]{index=0}
    
        # build penalties
        keys = list(S_list.keys())
        S_mono = lam_eff[-1] * S_list[keys[-1]]
        pen_mono = 0.5 * (c.T @ S_mono @ c)
    
        if len(X_basis) > 1:
            # non-mono exists → block-diag penalty in β-space
            S_non_dict = {k: S_list[k] for k in keys[:-1]}
            S_non = self.S_block_lambda(S_non_dict, lam_eff[:-1])
            pen_non = 0.5 * (beta_non.T @ S_non @ beta_non)
        else:
            pen_non = 0.0
    
        # unpenalized NLL still reparametrizes internally (as your current nll does)  :contentReference[oaicite:2]{index=2}
        return self.nll(beta_init, X_basis, dBxj) + pen_non + pen_mono

    def outer_optimization_obj_and_jac(self, log_lambda, X_basis, S_list, dBxj, k, beta_initial = None):
            
        """
        This function returns both the objective and the Jacobian of the outer
        optimization. Both are computed in one function to avoid repetition.
        """
            
        # =====================================================================
        # Step 1: find the optimal coefficients
        # =====================================================================
        beta_opt = self.find_optimal_coefficients(
            log_lambda, 
            X_basis, 
            S_list, 
            dBxj, 
            k,
            beta_initial = beta_initial)
    
        beta_opt = jnp.asarray(beta_opt)
    
        # ---------------------------------------------------------------------
        # CHANGED: make a reparametrized copy for places where the derivation
        #          requires beta in reparam space (KKT term and Ω_j beta RHS).
        #          Do NOT pass this into nll/hessian calls (they reparam internally).
        # ---------------------------------------------------------------------
        nbasis_mono = X_basis[-1].shape[1]
        beta_rep = jnp.asarray(beta_opt)
        beta_rep = beta_rep.at[-nbasis_mono:].set(self.reparametrize(beta_opt[-nbasis_mono:]))
    
        # =====================================================================
        # Step 2: Precompute objects
        # =====================================================================
        J_unpen = self.nll_jit(beta_opt, X_basis, dBxj)
        H_pen   = self.hessian_penalized_nll(beta_opt, log_lambda, X_basis, S_list, dBxj)
        H_unpen = self.hessian_nll(beta_opt, X_basis, dBxj)
        
        # Regularize and invert via linear solves
        p = beta_opt.size
        I = jnp.eye(p)
        eps = 1e-8 * jnp.mean(jnp.diag(H_pen))
        H_pen_reg = H_pen + eps * I
        H_pen_inv = jnp.linalg.solve(H_pen_reg, I)
    
        keys   = list(S_list.keys())
        blocks = [S_list[k_] for k_ in keys]
        sizes  = [blk.shape[0] for blk in blocks]
        starts = [0]
        for m in sizes[:-1]: starts.append(starts[-1] + m)
        ends   = [s + m for s, m in zip(starts, sizes)]
        q = len(blocks)
        lam_vec = jnp.atleast_1d(jnp.exp(log_lambda))
        
        # Block-wise scaling
        scale_vec = jnp.asarray([jnp.sqrt(jnp.trace(P.T @ P) / P.shape[0]) for P in X_basis])
        lam_eff_vec = lam_vec * (scale_vec ** 2)
    
        def Omega_lambda_mv(v):
            out = jnp.zeros_like(v)
            for j, Sj in enumerate(blocks):
                s, e = starts[j], ends[j]
                out = out.at[s:e].set(out[s:e] + lam_vec[j] * (Sj @ v[s:e]))
            return out
    
        # Reusable pieces
        X = H_pen_inv - (H_pen_inv @ H_unpen @ H_pen_inv)
        M = H_pen_inv @ H_unpen @ H_pen_inv
    
        # Use existing jitted gradient of unpenalized loss
        grad_unpen = lambda b: self.gradient_nll(b, X_basis, dBxj)
        
        mono_idx = q - 1
        s_m, e_m = starts[mono_idx], ends[mono_idx]
        f = lambda z: self.reparametrize(z)
        c_m, f_vjp = jax.vjp(f, beta_opt[s_m:e_m])
        jvp_mono   = lambda v: jax.jvp(f, (beta_opt[s_m:e_m],), (v,))[1]
        vjp_mono   = lambda v: f_vjp(v)[0]
        
        # full c-vector (reparam only on mono block; identity elsewhere)
        c_full = beta_opt.at[s_m:e_m].set(c_m)
    
        # =====================================================================
        # Step 3: Objective
        # =====================================================================
        
        # Effective dof for this component
        dof = jnp.trace(H_pen_inv @ H_unpen)
        
        # AICc/2 objective: J_unpen + dof + dof(dof+1)/(N-dof-1)
        N = jnp.asarray(X_basis[0].shape[0], dtype=J_unpen.dtype)
        denom = jnp.maximum(N - dof - 1.0, 1e-12)
        corr = dof * (dof + 1.0) / denom
        objective = J_unpen + dof + corr
        
        # d/d(dof) of the correction term, used to scale the dof-gradient
        corr_prime = ((2.0 * dof + 1.0) * denom + dof * (dof + 1.0)) / (denom ** 2)
        dof_mult = 1.0 + corr_prime
        
        # =====================================================================
        # Step 4: Jacobian w.r.t. log λ (per block)
        # =====================================================================
        jac_terms = []
    
        for j, Sj in enumerate(blocks):
            s, e = starts[j], ends[j]
            
            lamj = lam_eff_vec[j]
    
            wj = jnp.zeros_like(beta_rep)
            if j == mono_idx:
                wj = wj.at[s_m:e_m].set(lamj * vjp_mono(Sj @ c_m))
            else:
                wj = wj.at[s:e].set(lamj * (Sj @ beta_opt[s:e]))
            v_j = - H_pen_inv @ wj
    
    
            if j == mono_idx:
                def phi(z): 
                    cz = self.reparametrize(z)
                    return 0.5 * (cz @ (Sj @ cz))
                Hphi = hessian(phi)(beta_opt[s_m:e_m])
                direct_j = - jnp.trace(Hphi @ M[s_m:e_m, s_m:e_m])
            else:
                direct_j = - jnp.trace(Sj @ M[s:e, s:e])
            
            # Only the direct term gets the explicit λ_eff factor
            direct_j = lamj * direct_j
    
            # -----------------------------------------------------------------
            # KKT term
            # -----------------------------------------------------------------
            u = v_j
            u_m = jvp_mono(v_j[s_m:e_m]) 
            u = u.at[s_m:e_m].set(u_m)

            c_Omega_u = 0.0
            for i, Si in enumerate(blocks):
                si, ei = starts[i], ends[i]
                c_Omega_u = c_Omega_u + lam_eff_vec[i] * (c_full[si:ei] @ (Si @ u[si:ei]))
            kkt_j = - c_Omega_u 
            
            dG_apply_j = lambda x: jax.jvp(
                lambda bb: jax.jvp(grad_unpen, (bb,), (x,))[1],
                (beta_opt,), (v_j,)
            )[1]
            D_cols_j = jax.vmap(dG_apply_j, in_axes=0)(I.T)
            dtrace_dir_j = jnp.sum(X * D_cols_j.T)
            
            # Penalty Hessian β-dependence contributes for ALL j (H depends on β
            # only via mono block, but dβ*/dθ_j≠0 in general).
            def Hphi_of(z):
                def phi(z_):
                    cz = self.reparametrize(z_)
                    return 0.5 * (cz @ (Sj @ cz))
                return hessian(phi)(z)
            dHphi_dir = jax.jvp(Hphi_of, (beta_opt[s_m:e_m],), (v_j[s_m:e_m],))[1]
            dHpen_beta = jnp.zeros_like(H_pen).at[s_m:e_m, s_m:e_m].set(dHphi_dir)
            dtrace_penalty_j = jnp.trace(H_pen_inv @ (lam_eff_vec[mono_idx] * dHpen_beta) @ H_pen_inv @ H_unpen)
            dtrace_dir_j = dtrace_dir_j - dtrace_penalty_j

            # Total partial derivative
            dof_grad = direct_j + dtrace_dir_j
            jac_terms.append(kkt_j + dof_mult * dof_grad)
    
        # explicit dtypes/shapes for SciPy
        objective = float(objective)
        jac = np.asarray(jac_terms, dtype=float).reshape(len(blocks))
        return objective, jac
    
    def find_optimal_coefficients(self, log_lambda, X_basis, S_list, dBxj, k, beta_initial = None):
        
        # In the separable formulation, we only have to optimize the monotone
        # coefficients; the nonmonotone coefficients follow algebraically
        
        # =====================================================================
        # Step 1: Find the optimal monotone coefficients
        # =====================================================================
        
        # Extract the number of coefficients
        num_mon_coefficients = self.nbases[k][k]
        
        # Initiate beta variables with small non-zero values for monotonicity
        if beta_initial is None:
            beta_initial = jnp.zeros(num_mon_coefficients) + 1E-10
        else:
            beta_initial = beta_initial[k][-num_mon_coefficients:]
        
        # Solve the optimization problem for the monotone coefficients
        nll_opt = minimize(
            self.monotone_optimization_objective_jit, 
            beta_initial, 
            args=(X_basis, dBxj, log_lambda, S_list), 
            jac=self.monotone_optimization_objective_jac_jit, 
            hess=self.monotone_optimization_objective_hess_jit,
            method="Newton-CG",
            tol=1e-9,
        )
        
        # Extract the optimal monotone coefficients
        beta_mon_opt = nll_opt.x
        
        # =====================================================================
        # Step 2: Compute the optimal nonmonotone coefficients
        # =====================================================================
        
        # If there exist nonmonotone terms
        if len(X_basis) > 1: # Yes, there exist nonmonotone terms
        
            # Prepare some variables
            P_nonmon = jnp.hstack(X_basis[:-1]) # Nonmonotone basis function evaluations
            P_mon = X_basis[-1] # Monotone basis function evaluations
            
            # Use λ_eff = exp(log λ) * s_j^2 for non-mono blocks (consistent with outer objective)
            lam = jnp.exp(log_lambda)
            scale_vec = jnp.asarray([jnp.sqrt(jnp.trace(P.T @ P) / P.shape[0]) for P in X_basis])
            lam_eff = lam * (scale_vec ** 2)
            S_nonmon = self.S_block_lambda(
                {key: S_list[key] for key in list(S_list.keys())[:-1]},
                lam_eff[:-1]
            )
            
            A = P_nonmon.T @ P_nonmon + S_nonmon
            rid = 1e-6 * jnp.maximum(1.0, jnp.mean(jnp.diag(A)))
            A = A + rid * jnp.eye(A.shape[0])
            Mk = jnp.linalg.solve(A, P_nonmon.T)
            
            # Compute the optimal nonmonotone coefficients
            beta_nonmon_opt = -Mk @ P_mon @ self.reparametrize(beta_mon_opt)
            
            # Combine the coefficient vectors
            beta_opt = jnp.concatenate([beta_nonmon_opt, beta_mon_opt])
        
        else:
            
            # We have only the monotone beta coefficients
            beta_opt = beta_mon_opt
            
        # Return the optimal beta coefficients
        return beta_opt


    def effective_degrees_of_freedom(self, beta_opt, log_lambda, X_basis, S_list, dBxj):
        
        # Penalized and unpenalized Hessian
        H_pen   = self.hessian_penalized_nll(beta_opt, log_lambda, X_basis, S_list, dBxj)
        H_unpen = self.hessian_nll(beta_opt, X_basis, dBxj)
        # Regularize and avoid explicit inverse: trace(H^{-1}G) = trace( solve(H, G) )
        p = beta_opt.size
        I = jnp.eye(p)
        # ridge scaled to typical curvature to avoid near-singular solves
        eps = 1e-6 * jnp.maximum(1.0, jnp.mean(jnp.diag(H_pen)))
        H_pen_reg = H_pen + eps * I
        X = jnp.linalg.solve(H_pen_reg, H_unpen)
        return jnp.trace(X)
    
    def AICc(self, beta_opt, log_lambda, X_basis, S_list, dBxj, N=None):
        
        # Negative log-likelihood (unpenalized)
        negative_log_likelihood = self.nll_jit(beta_opt, X_basis, dBxj)
        
        # Effective degrees of freedom
        effective_dof = self.effective_degrees_of_freedom(beta_opt, log_lambda, X_basis, S_list, dBxj)
        
        # Sample size (per-component)
        if N is None:
            N = X_basis[0].shape[0]
        N = jnp.asarray(N, dtype=negative_log_likelihood.dtype)
        
        # AICc: AIC + 2*k*(k+1)/(N-k-1), using effective_dof as k
        denom = jnp.maximum(N - effective_dof - 1.0, 1e-12)
        correction = 2.0 * effective_dof * (effective_dof + 1.0) / denom
        
        return 2.0 * negative_log_likelihood + 2.0 * effective_dof + correction
    
    def AIC(self, beta_opt, log_lambda, X_basis, S_list, dBxj):
        return self.AICc(beta_opt, log_lambda, X_basis, S_list, dBxj)

    def create_jitted_functions(self):
        
        # Define the gradient and Hessian of the inner monotone optimization
        self.monotone_optimization_objective_jit = jit(self.monotone_optimization_objective)
        self.monotone_optimization_objective_jac_jit = jit(grad(self.monotone_optimization_objective))
        self.monotone_optimization_objective_hess_jit = jit(hessian(self.monotone_optimization_objective))

        # Define the gradient and Hessian of the unpenalized inner objective
        self.nll_jit = jit(self.nll)
        self.gradient_nll = jit(grad(self.nll))
        self.hessian_nll = jit(hessian(self.nll))

        # Define the gradient and Hessian of the penalized inner objective
        self.nll_penalized_jit = jit(self.nll_penalized)
        self.gradient_penalized_nll = jit(grad(self.nll_penalized))
        self.hessian_penalized_nll = jit(hessian(self.nll_penalized))
        
        return
    
    def vectorized_root_search_alternate(self, X, Zk, d, resolution = 1001, 
        skip_dimensions = 0):
        
        
        """
        This function is an alternative root search routine, not based on
        bisection but interpolation. 
        
        Only used for "monotone_optimization_objective_jac_jit monotonicity"
        
        Variables:
        
            X
                [array] : N-by-d array of samples inverted so far, where the 
                d-th column still contains the reference samples used as a 
                residual in the root finding process
                
            Zk
                [vector] : a vector containing the target values in the k-th 
                dimension, for which the root finding algorithm must solve.
                
            d
                [integer] : an integer variable defining what map component 
                is being evaluated. Corresponds to a dimension of sample space.        
                
        """
        
        import numpy as np
        import copy
        from scipy.interpolate import interp1d
        
        # Create a local copy
        X       = copy.copy(X)
        
        # ---------------------------------------------------------------------
        # Step 1: For separable monotonicity, all non-monotone terms are just
        # constant offsets. So let's just calculate that once
        
        beta_all = np.asarray(self.betas[d])
        nb = self.nbases[d][d]
        beta_mon = self.reparametrize(beta_all[-nb:])
        beta_reparam = np.concatenate([beta_all[:-nb], beta_mon])
        
        
        beta_nonmonotone    = beta_reparam[:-self.nbases[d][d]]
        beta_monotone       = beta_reparam[-self.nbases[d][d]:]
        
        # Inefficient, better do that in the inverse function -marked-
        X_basis_local = []
        for j in range(d): # We only need the basis function evaluations for the nonmonotone terms, so no d+1
            if self.sparsity[d - self.skip_dimensions,j] == 1: # Only do that for real dependencies
                X_basis_local.append(self.Bx[j](X[:,j]))
        
        # Evaluate the nonmonotone part
        if len(X_basis_local) > 0:
            offset = np.dot(
                np.hstack(X_basis_local),
                beta_nonmonotone[:,np.newaxis])[:,0]
        else:
            offset = 0
        
        # ---------------------------------------------------------------------
        # Step 2: Evaluate the forward map
        pts             = np.linspace(self.training_data_range[d][0],self.training_data_range[d][1],resolution)
        
        # Evaluate the basis functions for those pts
        interpolation_basis = self.Bx[d](pts)
        
        # Evaluate the monotone part
        out = np.dot(
            interpolation_basis,
            beta_monotone[:,np.newaxis])[:,0]
        
        # ---------------------------------------------------------------------
        # Step 3: Create a 1D interpolator
        itp     = interp1d(
            x           = out,
            y           = pts,
            fill_value  = "extrapolate")
        
        # ---------------------------------------------------------------------
        # Step 4: Evaluate the 1D interpolator
        
        # Find the target values
        target  = - offset + Zk
        
        # Find the target root
        result  = itp(target)
        
        # Save the result    
        X[:,d] = copy.copy(result)
        
        return X