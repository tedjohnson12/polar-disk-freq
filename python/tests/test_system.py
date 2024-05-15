"""
Tests for misaligned_cb_disk.system
"""
import rebound
import numpy as np
import pytest

from polar_disk_freq import system, params


def test_cross():
    a = np.array([1,0,0]).T
    b = np.array([1,0,0]).T
    assert np.all(system.cross(a,b).T == np.array([0,0,0]))
    
    a = np.array([1,0,0]).T
    b = np.array([0,1,0]).T
    assert np.all(system.cross(a,b).T == np.array([0,0,1]))
    
    a = np.array([1,0,0]).T
    b = np.array([0,0,1]).T
    assert np.all(system.cross(a,b).T == np.array([0,-1,0]))
    
    a = np.array([
        [0,1,0],
        [1,0,0]
    ]).T
    b = np.array([
        [0,1,0],
        [0,1,0]
    ]).T
    assert np.all(system.cross(a,b).T == np.array([[0,0,0],[0,0,1]]))

def test_dot():
    a = np.array([
        [0,1,0],
        [1,0,0]
    ]).T
    b = np.array([
        [0,1,0],
        [0,1,0]
    ]).T
    assert np.all(system.dot(a,b) == np.array([1,0]))
    
   

def test_system_init():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    assert sys.sim.particles['m1'].m == binary.mass1
    assert sys.sim.particles['m2'].m == binary.mass2
    assert sys.sim.particles['p'].m == planet.mass
    sys.sim.particles['p'].ax

def test_system_integerate():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    sys.integrate(np.linspace(0,30,15))

def test_system_integrate_orbits():
    binary = params.Binary(2,0.5,1,0)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    sys.integrate_orbits(10,1)

def test_properties():
    binary = params.Binary(2,0.5,1,0.)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    # sys.integrate_orbits(10,1)
    sys.integrate(np.linspace(0,0.2,5))
    
    assert ~np.any(np.isnan(sys.specific_angular_momentum)), 'failed for specific_angular_momentum'
    assert ~np.any(np.isnan(sys.specific_torque)), 'failed for specific_torque'
    assert ~np.any(np.isnan(sys.inclination)), 'failed for inclination'
    assert ~np.any(np.isnan(sys.inclination_dot)), 'failed for inclination_dot'
    assert ~np.any(np.isnan(sys.angular_momentum_binary)), 'failed for angular_momentum_binary'
    assert ~np.any(np.isnan(sys.z_hat)), 'failed for z_hat'
    assert np.all(sys.z_hat[2] == 1.), 'failed for z_hat value'
    assert ~np.any(np.isnan(sys.x_hat)), 'failed for x_hat'
    assert ~np.any(np.isnan(sys.lon_ascending_node)), 'failed for lon_ascending_node'
    0
    
def test_eccentricity_vector():
    ecc = 1e-6
    binary = params.Binary(2,0.5,1,ecc)
    planet = params.Planet(0,3,0,0,0,0,0)
    sim = rebound.Simulation()
    sys = system.System(binary,planet,sim)
    
    # sys.integrate_orbits(10,1)
    sys.integrate(np.linspace(0,3,5))
    e = sys.eccentricity_bin
    assert np.all(np.isclose(system.dot(e,e),ecc**2,atol=1e-6))
    x_hat = sys.x_hat
    assert np.all(np.isclose(x_hat[0],1.,atol=1e-6))
    y_hat = sys.y_hat
    assert np.all(np.isclose(y_hat[1],1.,atol=1e-6))
    z_hat = sys.z_hat
    assert np.all(np.isclose(system.cross(x_hat,y_hat),z_hat,atol=1e-6))
    assert ~np.any(np.isnan(sys.lon_ascending_node_dot)), 'failed for lon_ascending_node_dot'

    

if __name__ in '__main__':
    pytest.main(args=[__file__])