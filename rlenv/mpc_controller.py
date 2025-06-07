import numpy as np
import casadi as ca

class CassieMPC:
    """Model Predictive Controller for Cassie robot."""
    
    def __init__(self, horizon=10, dt=0.01, num_motors=10):
        """
        Initialize the MPC controller.
        
        Args:
            horizon: Prediction horizon length
            dt: Time step for prediction
            num_motors: Number of motors to control (10 for Cassie)
        """
        self.horizon = horizon
        self.dt = dt
        self.num_motors = num_motors
        
        # State dimensions (position, velocity for each motor + base pose)
        self.nx = 2 * num_motors + 6  # motor pos, motor vel, base pos (3), base rot (3)
        # Control dimensions (torque for each motor)
        self.nu = num_motors
        
        print(f"Initializing MPC with horizon={horizon}, dt={dt}, num_motors={num_motors}")
        
        # Setup optimization problem
        self._setup_dynamics_model()
        self._setup_optimization_problem()
        
        # Previous solution for warm starting
        self.prev_u_opt = None
        
    def _setup_dynamics_model(self):
        """Set up the dynamics model for MPC."""
        # State variables
        x = ca.SX.sym('x', self.nx)
        # Control variables
        u = ca.SX.sym('u', self.nu)
        
        # Extract state components
        motor_pos = x[:self.num_motors]
        motor_vel = x[self.num_motors:2*self.num_motors]
        base_pos = x[2*self.num_motors:2*self.num_motors+3]
        base_rot = x[2*self.num_motors+3:2*self.num_motors+6]
        
        # Ultra-conservative dynamics model focused primarily on stability
        # Much higher inertia values making the system respond more slowly
        motor_inertia = ca.SX([0.2, 0.2, 0.4, 0.4, 0.1, 0.2, 0.2, 0.4, 0.4, 0.1])
        # Very high damping for stability
        motor_damping = ca.SX([1.0, 1.0, 1.2, 1.2, 0.8, 1.0, 1.0, 1.2, 1.2, 0.8])
        
        # Compute accelerations (with damping)
        # Scale control input way down for very gentle actions
        control_scale = 0.2  # Very reduced control authority
        motor_acc = control_scale * u / motor_inertia - motor_damping * motor_vel / motor_inertia
        
        # Simple model for base movement based on leg configuration and velocity
        # In reality, this would be much more complex due to contact dynamics
        hip_pitch_idx = [2, 7]  # Indices for hip pitch joints
        knee_idx = [3, 8]       # Indices for knee joints
        
        # Minimal coupling between leg movements and base movements
        coupling_factor = 0.02  # Extremely small effect
        
        # Base dynamics - affected by leg movements (simplified)
        # Forward velocity influenced by hip pitch and knee angles - very minimal coupling
        base_vel_x = coupling_factor * (motor_vel[hip_pitch_idx[0]] + motor_vel[hip_pitch_idx[1]]) - 0.01 * (motor_vel[knee_idx[0]] + motor_vel[knee_idx[1]])
        
        # Lateral velocity influenced by hip roll angles
        hip_roll_idx = [0, 5]  # Indices for hip roll joints
        base_vel_y = 0.01 * (motor_vel[hip_roll_idx[0]] - motor_vel[hip_roll_idx[1]])
        
        # Vertical velocity - simplified model based on knee extension
        base_vel_z = -0.01 * (motor_vel[knee_idx[0]] + motor_vel[knee_idx[1]])
        
        # Angular velocity - simplified model for orientation changes
        # Minimal rotational effects
        base_omega_x = 0.02 * (motor_vel[hip_roll_idx[0]] - motor_vel[hip_roll_idx[1]])
        base_omega_y = 0.02 * (motor_vel[hip_pitch_idx[0]] - motor_vel[hip_pitch_idx[1]])
        base_omega_z = 0.01 * (motor_vel[hip_roll_idx[0]] - motor_vel[hip_roll_idx[1]])
        
        # State derivatives
        motor_pos_dot = motor_vel
        motor_vel_dot = motor_acc
        base_pos_dot = ca.vertcat(base_vel_x, base_vel_y, base_vel_z)
        base_rot_dot = ca.vertcat(base_omega_x, base_omega_y, base_omega_z)
        
        # Combine state derivatives
        x_dot = ca.vertcat(
            motor_pos_dot,
            motor_vel_dot,
            base_pos_dot,
            base_rot_dot
        )
        
        # Create discrete dynamics function (Euler integration)
        x_next = x + self.dt * x_dot
        
        # Create dynamics function
        self.dynamics = ca.Function('dynamics', [x, u], [x_next])
        
    def _setup_optimization_problem(self):
        """Set up the MPC optimization problem."""
        # Create optimization variables
        self.opt_x = []
        self.opt_x0 = []
        self.opt_lbx = []
        self.opt_ubx = []
        
        # Initial state
        x0 = ca.SX.sym('x0', self.nx)
        self.opt_x.append(x0)
        self.opt_x0.append(np.zeros(self.nx))
        self.opt_lbx.append(-np.inf * np.ones(self.nx))
        self.opt_ubx.append(np.inf * np.ones(self.nx))
        
        # Create optimization variables for each stage
        for k in range(self.horizon):
            # Control at stage k
            uk = ca.SX.sym(f'u_{k}', self.nu)
            self.opt_x.append(uk)
            self.opt_x0.append(np.zeros(self.nu))
            # Reduce torque limits for safety
            self.opt_lbx.append(-50 * np.ones(self.nu))  # Reduced from -100
            self.opt_ubx.append(50 * np.ones(self.nu))   # Reduced from 100
            
            # State at stage k+1
            xk1 = ca.SX.sym(f'x_{k+1}', self.nx)
            self.opt_x.append(xk1)
            self.opt_x0.append(np.zeros(self.nx))
            self.opt_lbx.append(-np.inf * np.ones(self.nx))
            self.opt_ubx.append(np.inf * np.ones(self.nx))
        
        # Concatenate optimization variables
        self.opt_x = ca.vertcat(*self.opt_x)
        self.opt_x0 = np.concatenate(self.opt_x0)
        self.opt_lbx = np.concatenate(self.opt_lbx)
        self.opt_ubx = np.concatenate(self.opt_ubx)
        
        # Create parameter for initial state and reference trajectory
        self.p = ca.SX.sym('p', self.nx + self.nx * (self.horizon + 1))
        
        # Extract initial state and reference trajectory
        x_init = self.p[:self.nx]
        x_ref = ca.reshape(self.p[self.nx:], self.nx, self.horizon + 1)
        
        # Create constraints
        self.g = []
        self.lbg = []
        self.ubg = []
        
        # Extract variables
        x_vars = []
        u_vars = []
        
        # Initial state constraint
        x_vars.append(self.opt_x[:self.nx])
        
        # Extract control and state variables
        for k in range(self.horizon):
            # Control at stage k
            u_vars.append(self.opt_x[self.nx + k*(self.nx + self.nu):self.nx + k*(self.nx + self.nu) + self.nu])
            
            # State at stage k+1
            x_vars.append(self.opt_x[self.nx + k*(self.nx + self.nu) + self.nu:self.nx + (k+1)*(self.nx + self.nu)])
        
        # Initial state constraint
        self.g.append(x_vars[0] - x_init)
        self.lbg.append(np.zeros(self.nx))
        self.ubg.append(np.zeros(self.nx))
        
        # Dynamics constraints
        for k in range(self.horizon):
            # Call dynamics function
            x_next = self.dynamics(x_vars[k], u_vars[k])
            
            # For CasADi, no need to flatten or reshape here as we're in symbolic mode
            self.g.append(x_vars[k+1] - x_next)
            self.lbg.append(np.zeros(self.nx))
            self.ubg.append(np.zeros(self.nx))
        
        # Concatenate constraints
        self.g = ca.vertcat(*self.g)
        self.lbg = np.concatenate(self.lbg)
        self.ubg = np.concatenate(self.ubg)
        
        # Create cost function
        cost = 0
        
        # Stage cost
        # Different weights for different state components - focus more on stability
        Q_diag = np.ones(self.nx)
        
        # Higher weights for velocities to reduce quick movements
        Q_diag[self.num_motors:2*self.num_motors] = 3.0  # Motor velocities - reduce velocity changes
        
        # Higher weights for base position and orientation - prioritize stability
        Q_diag[2*self.num_motors:2*self.num_motors+3] = 8.0  # Base position
        Q_diag[2*self.num_motors+3:] = 10.0  # Base orientation - highest priority
        
        Q = np.diag(Q_diag)
        
        # Control cost with different weights for different motors - penalize large torques more heavily
        R_diag = np.ones(self.nu) * 0.5  # Increased from 0.1 to penalize high control efforts
        
        # Higher weights for hip motors compared to foot motors (hip motions have more impact)
        R_diag[[0, 1, 5, 6]] *= 2.5  # Hip motors - increased from 2.0
        R = np.diag(R_diag)
        
        # Add control rate penalty to prevent rapid changes in torque
        control_rate_weight = 5.0  # Penalize rapid changes in control
        
        for k in range(self.horizon):
            # State error
            state_error = x_vars[k] - x_ref[:, k]
            
            # Control cost
            control_cost = ca.mtimes(ca.mtimes(u_vars[k].T, R), u_vars[k])
            
            # State cost
            state_cost = ca.mtimes(ca.mtimes(state_error.T, Q), state_error)
            
            # Control rate penalty (smooth control changes)
            if k > 0:
                control_rate = u_vars[k] - u_vars[k-1]
                control_rate_cost = control_rate_weight * ca.mtimes(control_rate.T, control_rate)
                cost += control_rate_cost
            
            # Add to total cost
            cost += control_cost + state_cost
        
        # Terminal cost - higher weight for stabilization at the end of horizon
        terminal_error = x_vars[-1] - x_ref[:, -1]
        # Increased from 10 to 20 for terminal cost
        terminal_cost = 20 * ca.mtimes(ca.mtimes(terminal_error.T, Q), terminal_error)
        cost += terminal_cost
        
        # Create NLP
        self.nlp = {
            'x': self.opt_x,
            'p': self.p,
            'f': cost,
            'g': self.g
        }
        
        # Set solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3,      # Increased tolerance for faster convergence
            'ipopt.acceptable_tol': 1e-2,  # Increased acceptable tolerance
            'ipopt.acceptable_iter': 3     # Accept solution after fewer iterations
        }
        
        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)
        
    def compute_control(self, state, reference_trajectory):
        """
        Compute optimal control action using MPC.
        
        Args:
            state: Current state [motor_pos, motor_vel, base_pos, base_rot]
            reference_trajectory: Reference trajectory for the full horizon [state_ref_0, state_ref_1, ..., state_ref_horizon]
                                 Should be of shape (nx, horizon+1)
            
        Returns:
            Optimal control action for the current time step
        """
        try:
            # Ensure state has the correct dimensions
            if len(state) != self.nx:
                print(f"Warning: State dimension mismatch. Expected {self.nx}, got {len(state)}. Truncating.")
                # Make sure state has correct dimensions for MPC model - truncate or pad if needed
                if len(state) > self.nx:
                    state = state[:self.nx]  # Truncate to expected size
                else:
                    # Pad with zeros if too short (shouldn't happen)
                    state = np.pad(state, (0, self.nx - len(state)), 'constant')
                    
            # Ensure reference trajectory has the correct shape
            if reference_trajectory.shape != (self.nx, self.horizon + 1):
                if reference_trajectory.shape[0] == self.nx:
                    # If only a single reference provided, repeat it for the entire horizon
                    reference_trajectory = np.tile(reference_trajectory.reshape(self.nx, 1), (1, self.horizon + 1))
                else:
                    raise ValueError(f"Reference trajectory shape must be ({self.nx}, {self.horizon + 1})")
            
            # Flatten the reference trajectory
            p = np.concatenate([state, reference_trajectory.flatten()])
            
            # Initialize solution if not available
            if self.prev_u_opt is None:
                self.prev_u_opt = np.zeros((self.horizon, self.nu))
            
            # Initialize solution with previous solution (warm start)
            # Shift previous solution by one step
            if self.prev_u_opt.shape[0] > 1:
                x0 = np.zeros(self.opt_x0.shape)
                
                # Set initial state
                x0[:self.nx] = state
                
                # Set controls (shift previous solution)
                for k in range(self.horizon-1):
                    idx_u = self.nx + k*(self.nx + self.nu)
                    x0[idx_u:idx_u + self.nu] = self.prev_u_opt[k+1]
                    
                # For the last control, repeat the last control from previous solution
                idx_u = self.nx + (self.horizon-1)*(self.nx + self.nu)
                x0[idx_u:idx_u + self.nu] = self.prev_u_opt[-1]
                
                # Simulate states based on the controls
                for k in range(self.horizon):
                    idx_x = self.nx + k*(self.nx + self.nu) + self.nu
                    if k > 0:
                        x_k = x0[idx_x - self.nx - self.nu:idx_x - self.nu]
                    else:
                        x_k = x0[:self.nx]
                    u_k = x0[self.nx + k*(self.nx + self.nu):self.nx + k*(self.nx + self.nu) + self.nu]
                    
                    try:
                        x_next = self.dynamics(x_k, u_k)
                    
                        # Convert from CasADi DM type to numpy array and ensure correct shape
                        if isinstance(x_next, ca.DM):
                            x_next = np.array(x_next).flatten()
                        else:
                            # Ensure array is 1D
                            x_next = np.array(x_next).flatten()
                        
                        x0[idx_x:idx_x + self.nx] = x_next
                    except Exception as inner_e:
                        print(f"Inner dynamics computation failed: {inner_e}")
                        # Just use the current state as next state
                        x0[idx_x:idx_x + self.nx] = x_k
            else:
                x0 = self.opt_x0
            
            # Solve optimization problem
            try:
                sol = self.solver(
                    x0=x0,
                    p=p,
                    lbx=self.opt_lbx,
                    ubx=self.opt_ubx,
                    lbg=self.lbg,
                    ubg=self.ubg
                )
                
                # Extract solution
                x_opt = np.array(sol['x'])
                
                # Extract optimal controls for the entire horizon
                u_opt = np.zeros((self.horizon, self.nu))
                for k in range(self.horizon):
                    idx_u = self.nx + k*(self.nx + self.nu)
                    u_opt_k = np.array(x_opt[idx_u:idx_u + self.nu])
                    # Explicitly ensure 1D array with correct shape
                    if len(u_opt_k.shape) > 1:
                        u_opt_k = u_opt_k.flatten()
                    u_opt[k] = u_opt_k
                    
                # Store solution for warm starting
                self.prev_u_opt = u_opt
                
                # Return only the first control action as a numpy array with proper shape
                result = np.array(u_opt[0]).flatten()
                return result
            
            except Exception as solver_e:
                print(f"MPC solver failed: {solver_e}")
                # Return a safe default action if optimization fails
                return np.zeros(self.nu)
                
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            # Return a safe default action if optimization fails
            return np.zeros(self.nu) 
            
    def reset(self):
        """Reset the MPC controller, clearing the previous solution."""
        self.prev_u_opt = None 