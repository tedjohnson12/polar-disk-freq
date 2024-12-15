"""
Code for calculating the precession timescale for :math:`j\\neq 0`.

"""
from typing import Tuple
from scipy.special import ellipk, ellipkinc
from scipy.optimize import newton
import numpy as np

from .rk4 import init_xyz, get_gamma, integrate

def get_h(eb:float,i0: float, omega0: float):
    """
    Farago & Laskar 2010 eq 2.20
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    i0 : float
        The inclination of the disk
    omega0 : float
        The longitude of the ascending node of the disk
    
    Returns
    -------
    float
    """
    lx, ly, lz = init_xyz(i0,omega0)
    return lz**2 - eb**2 * (4*lx**2 - ly**2)

def get_k_sq(eb:float,i0: float, omega0: float):
    """
    Farago & Laskar 2010 eq 2.31
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    i0 : float
        The inclination of the disk
    omega0 : float
        The longitude of the ascending node of the disk
    
    Returns
    -------
    float
    """
    h = get_h(eb,i0,omega0)
    return 5 * eb**2 / (1-eb**2) * (1-h)/(h+4*eb**2)

def elliptic_integral(k_sq: float):
    """
    Farago & Laskar 2010 eq 2.33
    
    Parameters
    ----------
    k_sq : float
        Defined in Farago & Laskar 2010 eq 2.31
    
    Returns
    -------
    float
    
    Notes
    -----
    The scipy implementation does not allow k > 1, so we have to do some rescaling. This has been tested and works.
    """
    if k_sq == 1:
        raise ValueError('k^2 = 1')
    if k_sq < 1:
        return ellipk(k_sq)
    else:
        # Note the scipy implementation does not allow k > 1, so we have to do this rescaling
        k = np.sqrt(k_sq)
        return 1/k * ellipkinc(np.arcsin(1),1/k)
    
def get_gamma_1(
    eb: float,
    i0: float,
    omega0:float
):
    """
    Last part of Farago & Laskar 2010 eq 2.32.
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    i0 : float
        The inclination of the disk
    omega0 : float
        The longitude of the ascending node of the disk
    
    Returns
    -------
    float
    """
    k_sq = get_k_sq(eb,i0,omega0)
    h = get_h(eb,i0,omega0)
    big_k = elliptic_integral(k_sq)
    return big_k * 1 / np.sqrt( (1-eb**2) * (h + 4*eb**2) )

def get_gamma_2(
    j: float,
    f: float,
    ab_ap: float,
    eb: float,
):
    """
    The $\\sqrt{1+m_r/M_b}$ term in Farago & Laskar 2010 eq 3.9.
    
    Converting this term into a form that involves $j$ is not easy. It involves finding
    the roots of a cubic polynomial. In the below function $\\beta$ is $\\sqrt{1+m_r/M_b}$
    and $d$ is a term that involves the parameters of the system. We solve for $\\beta$
    using the Newton-Raphson method.
    
    Parameters
    ----------
    j : float
        The angular momentum of the ring relative to the binary
    f : float
        The mass of the binary
    ab_ap : float
        The semimajor axis of the binary divided by the semimajor axis of the ring
    eb : float
        The eccentricity of the binary
    
    Returns
    -------
    float
    """
    d = j * f * (1-f) * np.sqrt(ab_ap * (1-eb**2))
    def func(beta: float) -> float:
        return beta**3 - beta - d
    guess = 1
    return newton(func,guess)

def get_tau_p(
    eb:float,
    j:float,
    i0: float,
    omega0: float,
    tau0: float,
    dtau0: float,
    walltime: float,
    epsilon: float
)-> Tuple[float, str]:
    """
    Get the precession timescale in units of $\\tau$.
    
    This function does a RKF integration and then returns the precession
    timescale along with the dynamical state of the system.
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    j : float
        The relative angular momentum of the disk relative to the binary
    i0 : float
        The inclination of the disk
    omega0 : float
        The longitude of the ascending node of the disk
    tau0 : float
        The initial time coordinate
    dtau0 : float
        The initial timestep
    walltime : float
        The time allowed for the integration, in seconds.
    epsilon : float
        The tolerance for the integration.
    
    Returns
    -------
    float
        The precession timescale in units of $\\tau$
    str
        The dynamical state of the system
    """
    x,y,z = init_xyz(i0,omega0)
    gamma = get_gamma(eb,j)
    _,_,_,_,_,tau_p,state = integrate(tau0,dtau0,x,y,z,eb,gamma,walltime,epsilon)
    
    return tau_p, state

def get_tp_over_tpj0(
    eb:float,
    j:float,
    i0: float,
    omega0:float,
    fb:float,
    ab_ap:float,
    tau0: float,
    dtau0: float,
    walltime: float,
    epsilon: float
)-> Tuple[float, str]:
    """
    Calculate the precession timescale relative to the timescale for a test particle.
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    j : float
        The relative angular momentum of the disk relative to the binary
    i0 : float
        The inclination of the disk
    omega0 : float
        The longitude of the ascending node of the disk
    fb : float
        The mass fraction of the binary
    ab_ap : float
        The semimajor axis of the binary divided by the semimajor axis of the ring
    tau0 : float
        The initial time coordinate
    dtau0 : float
        The initial timestep
    walltime : float
        The time allowed for the integration, in seconds.
    epsilon : float
        The tolerance for the integration.
    
    Returns
    -------
    float
        The precession timescale relative to the timescale for a test particle
    str
        The dynamical state of the system
    """
    tau,_state = get_tau_p(eb,j,i0,omega0,tau0,dtau0,walltime,epsilon)
    g1 = get_gamma_1(eb,i0,omega0)
    g2 = get_gamma_2(j, fb, ab_ap, eb)
    return tau/(4*g1*g2), _state