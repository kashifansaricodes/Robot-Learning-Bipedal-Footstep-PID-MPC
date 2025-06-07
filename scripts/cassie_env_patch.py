import numpy as np

def patch_cassie_env(env):
    """
    Apply patches to the CassieEnv class to fix issues with fall detection.
    """
    # Store the original is_done method
    original_is_done = env.__is_done
    
    # Define a new is_done method with enhanced fall detection
    def enhanced_is_done(self):
        # First check with the original method
        done = original_is_done()
        
        # Additional fall detection checks
        if not done:
            # Check for very low height
            if self.height < 0.6:  # If robot is too close to the ground
                self.fall_flag = True
                print(f"Enhanced fall detection: Robot too low (height={self.height:.3f})")
                return True
                
            # Check for extreme tilt angles
            if hasattr(self, 'curr_rpy_gt'):
                roll, pitch, _ = self.curr_rpy_gt
                if abs(roll) > 0.7 or abs(pitch) > 0.7:  # About 40 degrees
                    self.fall_flag = True
                    print(f"Enhanced fall detection: Robot too tilted (roll={roll:.3f}, pitch={pitch:.3f})")
                    return True
            
            # Check for high motor velocities (potential instability)
            if hasattr(self, 'qvel') and hasattr(self, 'motor_vel_idx'):
                motor_vels = self.qvel[self.motor_vel_idx]
                if np.any(np.abs(motor_vels) > 30):  # Threshold for motor velocities
                    self.fall_flag = True
                    print(f"Enhanced fall detection: Motors moving too fast (max vel={np.max(np.abs(motor_vels)):.3f})")
                    return True
        
        return done
    
    # Replace the is_done method
    env.__is_done = lambda: enhanced_is_done(env)
    
    print("Applied patches to CassieEnv for enhanced fall detection")
    return env 