"""
Parameters
----------

Control the input parameters of the rebound simulations.
"""
from numpy import pi, sqrt
from rebound import Simulation, Particle
import reboundx
from reboundx import constants

G = 1


def get_star_masses(mass_binary: float, mass_fraction: float):
    """
    Get the masses of two binary stars given
    the total mass of the binary :math:``M_b and the 
    mass fraction parameter :math:`f_b`

    Parameters
    ----------
    mass_binary : float
        The total mass of the binary
    mass_fraction : float
        The fraction :math:`M_2/M_b`

    Returns
    -------
    mass1 : float
        The mass of the primary star.
    mass2 : float
        The mass of the secondary star.
    """
    mass2 = mass_binary * mass_fraction
    mass1 = mass_binary*(1-mass_fraction)
    return mass1, mass2


class Binary:
    """
    Binary system parameters.

    Parameters
    ----------
    mass_binary : float
        The total mass of the binary.
    mass_fraction : float
        The mass fraction :math:`M_2/M_b`.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.

    Attributes
    ----------
    mass_binary : float
        The total mass of the binary.
    mass_fraction : float
        The mass fraction :math:`M_2/M_b`.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.
    name1 : str
        The identifier for the primary star.
    name2 : str
        The indentifier for the secondary star.
    mass1 : float
        The mass of the primary star.
    mass2 : float
        The mass of the secondary star.
    """
    name1 = 'm1'
    name2 = 'm2'

    def __init__(
        self,
        mass_binary: float,
        mass_fraction: float,
        semimajor_axis_binary: float,
        eccentricity_binary: float
    ):
        self.mass_binary = mass_binary
        self.mass_fraction = mass_fraction
        self.semimajor_axis_binary = semimajor_axis_binary
        self.eccentricity_binary = eccentricity_binary
    @property
    def period(self) -> float:
        """
        The period of the binary.

        :type: float
        """
        return 2*pi * sqrt(self.semimajor_axis_binary**3 / G / self.mass_binary)

    @property
    def mass1(self) -> float:
        """
        The mass of Star 1

        :type: float
        """
        mass1, _ = get_star_masses(self.mass_binary, self.mass_fraction)
        return mass1

    @property
    def mass2(self) -> float:
        """
        The mass of Star 2

        :type: float
        """
        _, mass2 = get_star_masses(self.mass_binary, self.mass_fraction)
        return mass2

    def add_to_sim(self, sim: Simulation, gr:bool):
        """
        Add binary system to a rebound Simulation.
        Then move to the CoM frame.

        Parameters
        ----------
        sim : rebound.Simulation
            The simulation to add the particles to.
        """
        star1 = Particle(
            simulation=sim,
            hash=self.name1,
            m=self.mass1
        )
        sim.add(star1)
        star2 = Particle(
            simulation=sim,
            a=self.semimajor_axis_binary,
            e=self.eccentricity_binary,
            hash=self.name2,
            m=self.mass2,
            primary=star1
        )
        sim.add(star2)
        sim.move_to_com()
        if gr:
            rebx = reboundx.Extras(sim)
            gx = rebx.load_force('gr')
            rebx.add_force(gx)
            gx.params["c"] = constants.C
            sim.particles[self.name1].params['gr_source'] = 1
            sim.particles[self.name2].params['gr_source'] = 1


class Planet:
    """
    Planet (as a disk proxy) parameters.

    Parameters
    ----------
    mass : float
        The mass of the planet.
    semimajor_axis : float
        The semimajor axis of the planet.
    inclination : float
        The inclination from the binary orbital plane.
    lon_ascending_node : float
        The longitude of the ascending node.
    true_anomaly : float
        The true anomaly to set as the initial state.
    eccentricity : float, optional
        The eccentricity of the planet's orbit, by default 0
    arg_pariapsis : float, optional
        The argument of pariapsis, by default 0
    """
    name = 'p'

    def __init__(
        self,
        mass: float,
        semimajor_axis: float,
        inclination: float,
        lon_ascending_node: float,
        true_anomaly: float,
        eccentricity: float = 0,
        arg_pariapsis: float = 0
    ):
        self.mass = mass
        self.semimajor_axis = semimajor_axis
        self.inclination = inclination
        self.lon_ascending_node = lon_ascending_node
        self.true_anomaly = true_anomaly
        self.eccentricity = eccentricity
        self.arg_pariapsis = arg_pariapsis

    def add_to_sim(self, sim: Simulation):
        """
        Add the planet to the rebound simulation.

        Parameters
        ----------
        sim : rebound.Simulation
            The simulation to add the particle to.
        """
        com = sim.com()
        planet = Particle(
            simulation=sim,
            primary=com,
            m=self.mass,
            a=self.semimajor_axis,
            inc=self.inclination,
            e=self.eccentricity,
            Omega=self.lon_ascending_node,
            omega=self.arg_pariapsis,
            f=self.true_anomaly,
            hash='p'
        )
        sim.add(planet)
        sim.move_to_com()
