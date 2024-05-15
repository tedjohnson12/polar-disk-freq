"""
Tests for params.py
"""
from rebound import Simulation, OrbitPlot
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import pytest

from polar_disk_freq import params


def test_star_mass():
    cases = (
        (1, 1, 0, 1),
        (1, 0, 1, 0),
        (2, 0.5, 1, 1),
        (0, 1, 0, 0),
        (0, 0, 0, 0)
    )
    for (mb, fb, m1, m2) in cases:
        _m1, _m2 = params.get_star_masses(mb, fb)
        assert _m1 == pytest.approx(m1, abs=1e-6), 'Failed for m1'
        assert _m2 == pytest.approx(m2, abs=1e-6), 'Failed for m2'


def test_Binary_star_masses():
    cases = (
        (1, 1, 0, 1),
        (1, 0, 1, 0),
        (2, 0.5, 1, 1),
        (0, 1, 0, 0),
        (0, 0, 0, 0)
    )
    for (mb, fb, m1, m2) in cases:
        binary = params.Binary(mb, fb, 0, 0)
        assert binary.mass1 == pytest.approx(m1, abs=1e-6), 'Failed for m1'
        assert binary.mass2 == pytest.approx(m2, abs=1e-6), 'Failed for m2'


def test_Binary_add_to_sim():
    binary = params.Binary(2, 0.5, 1, 0)
    sim = Simulation()
    binary.add_to_sim(sim,gr=False)
    primary = sim.particles['m1']
    secondary = sim.particles['m2']
    assert np.sqrt(
        (secondary.x-primary.x)**2
        + (secondary.y-primary.y)**2
        + (secondary.z-primary.z)**2
    ) == pytest.approx(1, abs=1e-6), 'Failed for m2 distance from m1'


def test_Binary_sim_orbit():
    outdir = Path(__file__).parent / 'output'
    outfile = outdir / 'binary_orbit.png'
    if not outdir.exists():
        outdir.mkdir()

    binary = params.Binary(2, 0.5, 1, 0)
    sim = Simulation()
    binary.add_to_sim(sim,gr=False)

    OrbitPlot(sim, unitlabel='[AU]').fig.savefig(outfile, facecolor='w')


def test_Planet_add_to_sim():
    binary = params.Binary(2, 0.5, 1, 0.2)
    sim = Simulation()
    binary.add_to_sim(sim, gr=False)
    planet = params.Planet(0, 3, 0, 0, 0, 0, 0)
    planet.add_to_sim(sim)
    pl = sim.particles['p']

    dist_from_com = np.sqrt(pl.x**2 + pl.y**2 + pl.z**2)
    assert dist_from_com == pytest.approx(3.0, abs=1e-6)

    total_time = 1*np.pi
    nsteps = 100
    time = np.linspace(0, total_time, nsteps)
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []
    for t in time:
        sim.integrate(t)
        m1 = sim.particles['m1']
        m2 = sim.particles['m2']
        p = sim.particles['p']
        x1.append(m1.x)
        y1.append(m1.y)
        x2.append(m2.x)
        y2.append(m2.y)
        x3.append(p.x)
        y3.append(p.y)
    outdir = Path(__file__).parent / 'output'
    outfile = outdir / 'planet_orbit.png'
    if not outdir.exists():
        outdir.mkdir()
    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.plot(x3, y3)
    ax.set_aspect('equal')
    fig.savefig(outfile, facecolor='w')


if __name__ in '__main__':
    test_Planet_add_to_sim()
