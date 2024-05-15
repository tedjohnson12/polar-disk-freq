"""
Test the rk4 solver
"""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from time import time

from polar_disk_freq.rk4 import get_gamma, init_xyz, integrate, get_i, get_omega


def test_get_gamma():
    eb = 0.5
    j = 0.0
    assert get_gamma(eb,j) == 0.0
    j = 1.0
    assert get_gamma(eb,j) == np.sqrt(1-0.25)

def test_init_xyz():
    i = 0.0
    omega = 0.0
    assert init_xyz(i,omega) == (0.0, 0.0, 1.0)
    i = np.pi/2
    omega = np.pi/2
    assert np.all(np.isclose(init_xyz(i,omega),(1.0, 0.0, 0.0),atol=1e-10))

def test_integration():
    i = np.pi * 0.4
    omega = np.pi/2
    dtau = 0.01
    lx_init, ly_init, lz_init = init_xyz(i,omega)
    eb_init = 0.4
    j = 0.1
    gamma = get_gamma(eb_init,j)
    
    tau,lx,ly,lz,eb,state = integrate(
        0.0, dtau, lx_init, ly_init, lz_init, eb_init, gamma, 10
    )
    assert state == 'l'
    
    omega_init = np.pi/2
    
    js = [1.0]
    incs = np.array([-0.99,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])*np.pi
    # incs = np.array([0.4])*np.pi

    colors = plt.cm.viridis(np.linspace(0,1,len(js)))
    time_int = 0
    for j,c in zip(js,colors):
        gamma = get_gamma(eb_init,j)
        for i in incs:
            lx_init, ly_init, lz_init = init_xyz(i,omega_init)
            start = time()
            tau,lx,ly,lz,eb,state = integrate(
                0.0, dtau, lx_init, ly_init, lz_init, eb_init, gamma, 10
            )
            end = time()
            time_int += end-start
            
            i = get_i(lx,ly,lz)
            omega = get_omega(lx,ly)
            plt.plot(i*np.cos(omega),i*np.sin(omega),c=c)
    print('Spent on integration:',time_int)
    print('Average:',time_int/len(incs))
    print('Time for 1000 runs:',time_int/len(incs)*1000)
    plt.axis('equal')
    plt.xlabel('$i\\cos(\\Omega)$')
    plt.ylabel('$i\\sin(\\Omega)$')
    0
    
if __name__ in '__main__':
    pytest.main(args=[__file__])