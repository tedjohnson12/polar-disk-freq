"""
Utilities for misaligned disks
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .system import System
from . import system

STATE_COLORS = {
    system.PROGRADE:'xkcd:crimson',
    system.RETROGRADE:'xkcd:azure',
    system.LIBRATING:'xkcd:violet',
    system.UNKNOWN:'xkcd:mint green'
}
STATE_LONG_NAMES = {
    system.PROGRADE:'Prograde',
    system.RETROGRADE:'Retrograde',
    system.LIBRATING:'Librating',
    system.UNKNOWN:'Unknown'
}

def phase_diag(sys:System,ax:Axes=None,**kwargs):
    """
    Plot the phase diagram in inclination - ascending node space.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.isinomega,**kwargs)

def phase_diag_cos(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`i\\cos{\\Omega}`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.icosomega,sys.icosomega_dot,**kwargs)
def phase_diag_sin(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`i\\sin{\\Omega}`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.isinomega,sys.isinomega_dot,**kwargs)

def lon_ascending_node_phase_diag(sys:System,ax:Axes=None,**kwargs):
    """
    A true phase diagram for :math:`\\Omega`, plotting it against its time derivative.
    
    Parameters
    ----------
    sys : System
        The system to plot.
    ax : Axes
        The axes to plot on.
    **kwargs
        Additional keyword arguments to pass to the plot.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(sys.lon_ascending_node,sys.lon_ascending_node_dot,**kwargs)