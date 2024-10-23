"""
Testing for system initialization.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import rebound
try:
    import polar_disk_freq
except ImportError as e:
    msg = 'You can install the polar_disk_freq package with:\npip install git+https://github.com/tedjohnson12/polar-disk-freq.git'
    raise ImportError(msg) from e
import numpy as np

OUTDIR = Path(__file__).parent / 'output'

J = [0,2]
FB = [0.5,0.7]

MB = 1
EB = 0.5
AB = 0.2
AP = 5*AB
INC = 0.9*np.pi


def get_mass_planet(
        _j: float,
        m_bin: float,
        f_bin: float,
        e_bin: float,
        a_bin: float,
        a_planet: float,
        e_planet: float
    ) -> float:
    """
    Get the mass of the planet given the binary eccentricity
    and the relative angular momentum.

    Parameters
    ----------
    _j : float
        The relative angular momentum.
    e_bin : float
        Binary eccentricity.

    Returns
    -------
    float
        The mass of the planet.

    """
    # From 2022MNRAS.517..732A eq 1,2,3
    if _j == 0:
        return 0
    a = 1
    b = m_bin
    c = 0
    d = -_j**2 * m_bin**3 * f_bin**2 * (1-f_bin)**2 * a_bin/a_planet * (1-e_bin**2)/(1-e_planet**2)
    
    roots = np.roots([a,b,c,d])
    roots = roots[np.isreal(roots) & (roots > 0)]
    assert roots.size == 1, f"Unexpected number of roots: {roots}"
    return roots[0]


def setup(i:float,e:float, fb:float,_j:float)->polar_disk_freq.system.System:
    mp = get_mass_planet(
        _j=_j,
        m_bin=MB,
        f_bin=fb,
        e_bin=EB,
        a_bin=AB,
        a_planet=AP,
        e_planet=0
    )
    binary = polar_disk_freq.params.Binary(MB,fb,AB,e)
    planet = polar_disk_freq.params.Planet(mp,AP,i,np.pi/2,0,0,0)
    sim = rebound.Simulation()
    _sys = polar_disk_freq.system.System(binary,planet,gr=False,sim=sim)
    return _sys

if __name__ in '__main__':
    header = ['j','fb','i','|h|^2','|r|^2','|v|^2','r.v']
    print('\t'.join(header))
    for j in J:
        for fb in FB:
            sys = setup(i=INC,e=EB,fb=fb,_j=j)
            h = sys.specific_angular_momentum
            r = sys.rp-sys.r_bin_com
            v = sys.rp_dot-sys.r_bin_com_dot
            row = [
                j, fb, sys.inclination[0],
                polar_disk_freq.system.dot(h,h)[0],
                polar_disk_freq.system.dot(r,r)[0],
                polar_disk_freq.system.dot(v,v)[0],
                polar_disk_freq.system.dot(r,v)[0]
            ]
            print('\t'.join(f'{r:.3f}' for r in row))
    js = np.linspace(0,2,21)
    hmags = []
    for j in js:
        sys = setup(i=INC,e=EB,fb=0.5,_j=j)
        h = sys.specific_angular_momentum
        hmags.append(polar_disk_freq.system.dot(h,h)[0])
    plt.plot(js,hmags)
    plt.savefig(OUTDIR / 'h_mag.png',dpi=300)
            