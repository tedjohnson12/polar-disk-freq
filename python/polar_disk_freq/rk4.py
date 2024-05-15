"""
Python wrapper for bindings
"""

from typing import Tuple
import numpy as np

from . import _polar_disk_freq


def get_gamma(eb:float,j:float)->float:
    """
    Get the constant of motion :math:`\\gamma = \\sqrt(1 - e_b^2) j`.
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    j : float
        The relative angular momentum of the planet relative to the binary
    
    Returns
    -------
    float
        The constant of motion :math:`\\gamma`
    """
    return _polar_disk_freq.get_gamma(eb,j)

def init_xyz(
    i:float,
    omega:float
) -> Tuple[float, float, float]:
    """
    Initialize the tilt vector give the inclination and longitude of the ascending node.
    
    Parameters
    ----------
    i : float
        The inclination
    omega : float
        The longitude of the ascending node
    
    Returns
    -------
    Tuple[float, float, float]
        The x, y, and z components of the tilt vector.
    """
    return _polar_disk_freq.init_xyz(i,omega)

def integrate(
    tau_init:float,
    dtau:float,
    lx_init:float,
    ly_init:float,
    lz_init:float,
    eb_init:float,
    gamma:float,
    walltime:float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Integrate equations (7-10) from Martin & Lubow (2019).
    
    Parameters
    ----------
    tau_init : float
        The initial time coordinate
    dtau : float
        The time step
    lx_init : float
        The initial x component of the tilt vector
    ly_init : float
        The initial y component of the tilt vector
    lz_init : float
        The initial z component of the tilt vector
    eb_init : float
        The initial eccentricity
    gamma : float
        The constant of motion :math:`\\gamma`
    walltime : float
        The total integration time allowed in seconds.
    
    Returns
    -------
    tau : np.ndarray
        The time coordinate
    lx : np.ndarray
        The x component of the tilt vector
    ly : np.ndarray
        The y component of the tilt vector
    lz : np.ndarray
        The z component of the tilt vector
    state : str
        The state of the system
    """
    return _polar_disk_freq.integrate_py(tau_init, dtau, lx_init, ly_init, lz_init, eb_init, gamma, walltime)

def get_i(
    lx:np.ndarray,
    ly:np.ndarray,
    lz:np.ndarray
) -> np.ndarray:
    """
    Compute the inclination from the tilt vector components.
    
    Parameters
    ----------
    lx : np.ndarray
        The x component of the tilt vector
    ly : np.ndarray
        The y component of the tilt vector
    lz : np.ndarray
        The z component of the tilt vector
    
    Returns
    -------
    np.ndarray
        The inclination
    """
    return np.arccos(lz/np.sqrt(lx**2+ly**2+lz**2))

def get_omega(
    lx:np.ndarray,
    ly:np.ndarray,
):
    """
    Compute the longitude of the ascending node from the tilt vector components.
    
    Parameters
    ----------
    lx : np.ndarray
        The x component of the tilt vector
    ly : np.ndarray
        The y component of the tilt vector
    
    Returns
    -------
    np.ndarray
        The longitude of the ascending node
    """
    return np.unwrap(np.arctan2(lx,-ly))