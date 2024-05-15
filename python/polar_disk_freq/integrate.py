"""
Module to find the area inside of the parameterized equation:

.. math::
    x = i \\cos{\\Omega} \\\\
    y = i \\sin{\\Omega}

For one procession of the orbit.
"""
import numpy as np

def cast_r_and_theta(r: np.ndarray, theta: np.ndarray):
    """
    Reorder the polar coordinates r and theta so that
    the discontinuies at pi and -pi are at the bounds
    of integration.

    Parameters
    ----------
    r : np.ndarray
        The polar coordinate r.
    theta : np.ndarray
        The polar coordinate theta.

    Returns
    -------
    r : np.ndarray
        The polar coordinate r, reordered.
    theta : np.ndarray
        The polar coordinate theta, reordered.
    """
    i_discontinuity = np.argmax(np.abs(np.diff(theta)))
    before = slice(0, i_discontinuity+1)
    after = slice(i_discontinuity+1, None)
    theta = np.concatenate((theta[after], theta[before]))
    r = np.concatenate((r[after], r[before]))
    return r, theta


def integrate_r_and_theta(
    r: np.ndarray,
    theta: np.ndarray
):
    """
    Use the trapezoidal rule to integrate r and theta.

    Parameters
    ----------
    r : np.ndarray
        The polar coordinate r.
    theta : np.ndarray
        The polar coordinate theta.

    Returns
    -------
    float
        The integral of cos(r)-1 as a function of theta.
    """
    return np.trapz(np.cos(r)-1, theta)
