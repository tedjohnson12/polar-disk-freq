"""
Analytic solutions
"""

import numpy as np
from scipy.integrate import simpson

class Zanazzi2018:
    """
    bibcode 2018MNRAS.473..603Z
    """
    citet = 'Zanazzi & Lai (2018)'
    @staticmethod
    def i_crit(ecc_b):
        return np.arctan(np.sqrt((1-ecc_b**2)/(5*ecc_b**2)))
    @staticmethod
    def omega_min(inclination: float, ecc_b: float) -> float:
        """
        The minimum true anomaly for a given inclination and eccentricity.

        Parameters
        ----------
        inclination : float
            The inclination
        eccentricity : float
            The eccentricity

        Returns
        -------
        float
            The minimum true anomaly
        """
        if np.abs(np.sin(inclination)) < np.abs(np.sin(Zanazzi2018.i_crit(ecc_b))):
            return np.pi/2
        else:
            return np.arcsin(np.tan(Zanazzi2018.i_crit(ecc_b))/np.abs(np.tan(inclination)))
    @staticmethod
    def prob_polar(inclination, ecc_b):
        return 1 - 2*Zanazzi2018.omega_min(inclination, ecc_b)/np.pi
    @staticmethod
    def frac_polar(ecc_b:float, n_points:int=2047):
        if n_points%2 == 0:
            raise ValueError('n_points must be odd')
        inclination = np.linspace(0, np.pi, n_points)
        jacobian = np.sin(inclination)
        prob_polar = np.array([Zanazzi2018.prob_polar(i, ecc_b) for i in inclination])
        numerator = simpson(y=prob_polar*jacobian,x=inclination)
        denominator = simpson(y=jacobian,x=inclination)
        return numerator/denominator
        
        