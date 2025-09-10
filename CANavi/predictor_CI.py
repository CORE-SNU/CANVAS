import os
import numpy as np

class Predictor_CI:
    def __init__(self,r_star=[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3]):
        self.r_star = r_star

    def CI_default(self,intervals):
        shifts= self.CI_default_MX(intervals)
        radius = np.nanmean(shifts)
        return radius
    def CI_default_MX(self,intervals):
        shifts= (np.array(self.r_star) - intervals) / np.array(self.r_star)
        return shifts