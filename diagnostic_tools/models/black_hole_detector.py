import numpy as np
from scipy.optimize import minimize
from astropy.constants import G, c

class BlackHoleDetector:
    def __init__(self, star_motion_data: np.ndarray):
        self.star_motion_data = star_motion_data

    def detect_black_hole(self):
        # Define the objective function to minimize
        def objective(params):
            M, a = params
            return np.sum((self.star_motion_data - self.calculate_star_motion(M, a)) ** 2)

        # Define the function to calculate the star motion
        def calculate_star_motion(M, a):
            return np.sqrt(G * M / (a * c ** 2))

        # Initialize the parameters
        M_init = 1e30
        a_init = 1e10

        # Minimize the objective function
        res = minimize(objective, [M_init, a_init])

        # Extract the optimized parameters
        M_opt, a_opt = res.x

        return M_opt, a_opt
