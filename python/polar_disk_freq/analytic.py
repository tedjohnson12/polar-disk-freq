"""
Analytic solutions
"""

from typing import Callable
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
        

class MartinLubow2019:
    """
    bibcode 2019MNRAS.490.1332M
    """
    citet = 'Martin & Lubow (2019)'
    
    @staticmethod
    def omega_min_1(
        j: float,
        e_b:float,
        i: float
    ) -> float:
        """
        The minimum longitude of the ascending node to satisfy :math:`\\Lambda_1 > 0`
        in Eq 32.
        
        According to the first paragraph of Sec 3.3.2, this is valid when :math:`\\chi \\ge 0`
        """
        return np.arcsin(
            np.sqrt(
                (1-e_b**2)/(5*e_b**2)*(2*j+np.cos(i))**2/np.sin(i)**2
            )
        )

    @staticmethod
    def chi(
        j: float,
        e_b:float,
        i: float
    ) -> float:
        """
        :math:`chi` is a constant of motion given by Eq 31
        """
        return e_b**2 - 2*(1-e_b**2)*j*(2*j+np.cos(i))
    
    @staticmethod
    def omega_min_2(
        j: float,
        e_b:float,
        i: float
    ) -> float:
        """
        The minimum longitude of the ascending node to satisfy :math:`\\Lambda_2 > 0`
        in Eq 38.
        
        This is valid when :math:`\\chi < 0`
        
        """
        sini = np.sin(i)
        
        a = 2/(5 * sini**2)
        b = np.cos(i) / (5 * j * sini**2)
        c = e_b**2 / ( 20 * j**2 * (1 - e_b**2) * sini**2)
        
        sin_omega_sq = a + b - c
        return np.arcsin(np.sqrt(sin_omega_sq))
    
    @staticmethod
    def omega_min(
        j: float,
        e_b:float,
        i: float
    ) -> float:
        """
        The minimum longitude of the ascending node that will lead to a polar orbit.
        
        Parameters
        ----------
        j : float
            The relative angular momentum
        e_b : float
            The eccentricity of the binary
        i : float
            The mutual inclination
            
        Returns
        -------
        float
        """
        if MartinLubow2019.is_only_circulating(j, e_b, i):
            return np.pi/2
        chi = MartinLubow2019.chi(j, e_b, i)
        if chi >= 0:
            omega =  MartinLubow2019.omega_min_1(j, e_b, i)
        else:
            omega =  MartinLubow2019.omega_min_2(j, e_b, i)
        return omega
    
    @staticmethod
    def is_only_circulating(j, e_b, i) -> bool:
        """
        True if only circulating orbits are possible regardless of the
        longitude of the ascending node.
        """
        chi = MartinLubow2019.chi(j, e_b, i)
        if chi >= 0:
            lhs = (5*e_b**2)/(1-e_b**2) * np.sin(i)**2 - (np.cos(i) + 2*j)**2
        else:
            lhs = np.sin(i)**2 - np.cos(i)/(5*j) + e_b**2/(20*j**2*(1-e_b**2)) - 2/5
        return lhs < 0

            
    @staticmethod
    def prob_polar(j, e_b, i):
        """
        Polar probability assuming that omega is evenly distributed.
        """
        return 1 - 2*MartinLubow2019.omega_min(j, e_b, i)/np.pi
    @staticmethod
    def frac_polar(
        j:float,
        e_b:float,
        n_points:int=2047,
        jacobian: Callable[[np.ndarray], np.ndarray] = lambda x: 0.5*np.sin(x)
        )->float:
        """
        Compute the fraction of polar orbits assuming the disk is oriented
        isotropically.
        
        Parameters
        ----------
        j : float
            The relative angular momentum
        e_b : float
            The eccentricity of the binary
        n_points : int
            The number of points to use in the integration
        
        Returns
        -------
        float
        """
        if n_points%2 == 0:
            raise ValueError('n_points must be odd')
        inclination = np.linspace(0, np.pi, n_points)
        jac = jacobian(inclination)
        prob_polar = np.array([MartinLubow2019.prob_polar(j, e_b, i) for i in inclination])
        return simpson(y=prob_polar*jac,x=inclination)