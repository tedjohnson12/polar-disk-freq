"""

The maximum allowed value of j for a disk before it becomes gravitationally unstable.
"""
from numpy import sqrt

def jmax_p_less_than_2(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    alpha is r_in/ab, beta is r_out/r_in
    """
    coeff = 2 * h / (
        f * (1-f)*(5/2-p) * sqrt(1-eb**2)
    )
    r_in_term = sqrt(alpha)
    r_out_term = (beta**(5/2-p) - 1) * beta**(p-2)
    return coeff * r_in_term * r_out_term

def jmax_p_greater_than_2(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    alpha is r_in/ab, beta is r_out/r_in
    """
    coeff = 2 * h / (
        f * (1-f)*(5/2-p) * sqrt(1-eb**2)
    )
    r_in_term = sqrt(alpha)
    r_out_term2 = beta**(5/2-p) - 1
    return coeff * r_in_term * r_out_term2

def jmax(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    The maximum value of j that does not result in gravitational instability.
    
    Parameters
    ----------
    h: float
        The disk aspect ratio
    f: float
        The binary mass fraction
    p: float
        The power law index
    eb: float
        The eccentricity of the binary
    alpha: float
        The inner radius divided by the binary semimajor axis
    beta: float
        The outer radius divided by the inner radius
    
    Returns
    -------
    float
        The maximum value of j
    """
    if p <=2:
        return jmax_p_less_than_2(h,f,p,eb,alpha,beta)
    else:
        return jmax_p_greater_than_2(h,f,p,eb,alpha,beta)
