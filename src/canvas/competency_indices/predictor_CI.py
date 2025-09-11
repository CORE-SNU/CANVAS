import os
import numpy as np

class Predictor_CI:
    def __init__(self,r_star=[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3]):
        """Predictor wrapper for different predictor CI(Competency Index).
        Args:
            r_star: The acceptable radius for each prediction step.
"""
        self.r_star = r_star

    def CI_default(self,intervals):
        """Calculate the default CI based on r_star and given intervals and average across all shifts."""
        shifts= self.CI_default_MX(intervals)
        radius = np.nanmean(shifts)
        return radius
    def CI_default_MX(self,intervals):
        """Calculate the default CI based on r_star and given intervals and return the difference at each timestep."""
        shifts= (np.array(self.r_star) - intervals) / np.array(self.r_star)
        return shifts