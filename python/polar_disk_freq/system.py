"""
Circumbinary system module.
"""

import warnings
from typing import Callable, Iterable, Tuple, Union
import numpy as np
import rebound
from tqdm.auto import tqdm

from . import params
from .integrate import integrate_r_and_theta, cast_r_and_theta

UNKNOWN = 'u'
"""
The state of the system is unknown.

:type: str
"""
LIBRATING = 'l'
"""
The system is in a librating mode.

:type: str
"""
PROGRADE = 'p'
"""
The system is prograde.

:type: str
"""
RETROGRADE = 'r'
"""
The system is retrograde.

:type: str
"""

def central_difference(x,y,n:int)->Tuple[np.ndarray,np.ndarray]:
    """
    Take the derivative using central differences
    """
    dy = y[2*n:] - y[:-2*n]
    dx = x[2*n:] - x[:-2*n]
    _x = x[n:-n]
    _y = dy/dx
    assert len(_x) == len(_y), f'{len(_x)} != {len(_y)}'
    return x[n:-n], dy/dx

def get_j(
    mb: float,
    fb: float,
    ab: float,
    eb: float,
    mp: float,
    ap: float,
)->float:
    """
    Compute the relative angular momentum of the planet relative to the binary.
    Eq 6 from Abod et al 2022
    
    Parameters
    ----------
    mb : float
        The mass of the binary
    fb : float
        The mass fraction of the binary
    ab : float
        The semimajor axis of the binary
    eb : float
        The eccentricity of the binary
    mp : float
        The mass of the planet
    ap : float
        The semimajor axis of the planet
    
    Returns
    -------
    float
        The relative angular momentum of the planet relative to the binary.
    """
    num = mp/mb * np.sqrt(ap/ab) * np.sqrt(1+mp/mb)
    den = fb * (1-fb) * np.sqrt(1-eb**2)
    return num/den

def get_j_crit(eb: float)->float:
    """
    Compute the critical value of the relative angular momentum of the planet relative to the binary.
    Eq 5 from Abod et al 2022
    
    Parameters
    ----------
    eb : float
        The eccentricity of the binary
    
    Returns
    -------
    float
        The critical value of the relative angular momentum of the planet relative to the binary.
    """
    return (1+4*eb**2) / (2 + 3*eb**2)

def get_wrapper(desc, total, start=0):
    """
    Returns a wrapper function that wraps an iterable object with a tqdm progress bar.

    Parameters
    ----------
    desc : str
        A string description of the progress bar.
    total : int
        The total number of iterations.
    start : int, optional
        The initial value of the progress bar. Defaults to 0.

    Returns
    -------
    wrapper : function
        A wrapper function that takes an iterable object and returns a tqdm progress bar.
    """
    def wrapper(_iter: Iterable) -> Iterable:
        """
        A function that wraps an iterable object with a tqdm progress bar.

        Parameters
        ----------
        _iter : Iterable
            The iterable object to be wrapped.

        Returns
        -------
        Iterable
            The wrapped iterable object with a progress bar.
        """
        return tqdm(_iter, desc=desc, total=total, initial=start, leave=False)
    return wrapper


def cross(a: np.ndarray, b: np.ndarray):
    """
    Compute the cross product

    Parameters
    ----------
    a : np.ndarray, shape=(3,N)
        The first vector argument.
    b : np.ndarray, shape=(3,N)
        The second vector argument.

    Returns
    -------
    np.ndarray, shape=(3,N)
        The cross product of a and b.
    """
    return np.cross(a, b, axisa=0, axisb=0).T


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the dot product.

    Parameters
    ----------
    a : np.ndarray, shape=(3,N)
        The first vector argument.
    b : np.ndarray, shape=(3,N)
        The second vector argument.

    Returns
    -------
    np.ndarray, shape=(3,N)
        The cross product of a and b.
    """
    return np.einsum('ij,ij->j', a, b)


class System:
    """
    A circumbinary system.

    Parameters
    ----------
    binary : params.Binary
        The binary star parameters.
    planet : params.Planet
        The planetary parameters.
    gr : bool
        Whether or not to include General Relativity.

    Attributes
    ----------
    binary : params.Binary
        The binary star parameters.
    planet : params.Planet
        The planetary parameters.
    gr : bool
        Whether or not to include General Relativity.
    sim : rebound.Simulation
            The simulation to add the particles to, by default None
    """
    _init_shape = (3, 0)
    _ix = 0
    _iy = 1
    _iz = 2
    _integrator = 'ias15'

    def __init__(
        self,
        binary: params.Binary,
        planet: params.Planet,
        gr: bool,
        sim: Union[rebound.Simulation,None] = None
    ):
        self.binary = binary
        self.planet = planet
        self.gr = gr
        if sim is None:
            sim = rebound.Simulation()
        self.sim = sim
        self._add_to_sim()

        # setup
        self.r1 = np.zeros(shape=self._init_shape)
        self.r1_dot = np.zeros(shape=self._init_shape)
        self.r1_2dot = np.zeros(shape=self._init_shape)
        self.r2 = np.zeros(shape=self._init_shape)
        self.r2_dot = np.zeros(shape=self._init_shape)
        self.r2_2dot = np.zeros(shape=self._init_shape)
        self.rp = np.zeros(shape=self._init_shape)
        self.rp_dot = np.zeros(shape=self._init_shape)
        self.rp_2dot = np.zeros(shape=self._init_shape)
        self.t = np.zeros(shape=(1, 0))
        self.integrate(times=np.array([0]), verbose=0, wrapper=None)

    @classmethod
    def from_params(
        cls,
        mass_binary: float,
        mass_fraction: float,
        semimajor_axis_binary: float,
        eccentricity_binary: float,
        mass_planet: float,
        semimajor_axis_planet: float,
        inclination_planet: float,
        lon_ascending_node_planet: float,
        true_anomaly_planet: float,
        gr: bool,
        eccentricity_planet: float = 0,
        arg_pariapsis_planet: float = 0,
        sim: rebound.Simulation = None
    ):
        """
        Generate a System object from parameters.

        Parameters
        ----------
        mass_binary : float
            The mass of the binary.
        mass_fraction : float
            The mass fraction :math:`M_2/M_b`.
        semimajor_axis_binary : float
            The semimajor axis of the binary.
        eccentricity_binary : float
            The eccentricity of the binary.
        mass_planet : float
            The mass of the planet.
        semimajor_axis_planet : float
            The semimajor axis of the planet.
        inclination_planet : float
            The inclination from the binary orbital plane.
        lon_ascending_node_planet : float
            The longitude of the ascending node.
        true_anomaly_planet : float
            The true anomaly to set as the initial state.
        eccentricity_planet : float, optional
            The eccentricity of the planet's orbit, by default 0
        arg_pariapsis_planet : float, optional
            The argument of pariapsis, by default 0,
        sim : rebound.Simulation
            The simulation to add the particles to, by default None

        """
        binary = params.Binary(
            mass_binary=mass_binary,
            mass_fraction=mass_fraction,
            semimajor_axis_binary=semimajor_axis_binary,
            eccentricity_binary=eccentricity_binary
        )
        planet = params.Planet(
            mass=mass_planet,
            semimajor_axis=semimajor_axis_planet,
            inclination=inclination_planet,
            lon_ascending_node=lon_ascending_node_planet,
            true_anomaly=true_anomaly_planet,
            eccentricity=eccentricity_planet,
            arg_pariapsis=arg_pariapsis_planet
        )
        return cls(binary, planet, gr, sim)

    def _add_to_sim(self):
        """
        Add the particles to the simulation.

        :type: NoneType
        """
        self.binary.add_to_sim(self.sim,self.gr)
        self.planet.add_to_sim(self.sim)
        self.sim.integrator = self._integrator

    def integrate(
        self,
        times: np.ndarray,
        verbose: int = 1,
        wrapper: Callable = None
    ):
        """
        Integrates the system over a given set of time steps.

        Parameters
        ----------
        times : np.ndarray
            An array of time steps at which to integrate the system.
        verbose : int, optional
            The level of verbosity during integration. Defaults to 1.
        wrapper : Callable, optional
            A function to wrap the iterator during integration. Defaults to None.

        Returns
        -------
        None
        """
        if not times.ndim == 1:
            raise ValueError('times must be 1d')
        n_steps = times.shape[0]
        shape = (3, n_steps)
        r1 = np.empty(shape)
        r1_dot = np.empty(shape)
        r1_2dot = np.empty(shape)
        r2 = np.empty(shape)
        r2_dot = np.empty(shape)
        r2_2dot = np.empty(shape)
        rp = np.empty(shape)
        rp_dot = np.empty(shape)
        rp_2dot = np.empty(shape)

        def wrap(it) -> Iterable:
            """
            Wrap the given iterable with a progress bar if verbose is set to 1, 
            otherwise return the iterable as is.

            Parameters
            ----------
            it : iterable
                The iterable to be wrapped.

            Returns
            -------
            iterable
                The wrapped iterable with a progress bar if verbose is
                set to 1, otherwise the original iterable.
            """
            if verbose == 1:
                return tqdm(it, desc='integrating', total=n_steps)
            else:
                return it
        if wrapper is None:
            wrapper = wrap
        for i, time in wrapper(enumerate(times)):
            self.sim.integrate(time)
            name = self.binary.name1
            r1[self._ix, i] = self.sim.particles[name].x
            r1[self._iy, i] = self.sim.particles[name].y
            r1[self._iz, i] = self.sim.particles[name].z
            r1_dot[self._ix, i] = self.sim.particles[name].vx
            r1_dot[self._iy, i] = self.sim.particles[name].vy
            r1_dot[self._iz, i] = self.sim.particles[name].vz
            r1_2dot[self._ix, i] = self.sim.particles[name].ax
            r1_2dot[self._iy, i] = self.sim.particles[name].ay
            r1_2dot[self._iz, i] = self.sim.particles[name].az
            # M2
            name = self.binary.name2
            r2[self._ix, i] = self.sim.particles[name].x
            r2[self._iy, i] = self.sim.particles[name].y
            r2[self._iz, i] = self.sim.particles[name].z
            r2_dot[self._ix, i] = self.sim.particles[name].vx
            r2_dot[self._iy, i] = self.sim.particles[name].vy
            r2_dot[self._iz, i] = self.sim.particles[name].vz
            r2_2dot[self._ix, i] = self.sim.particles[name].ax
            r2_2dot[self._iy, i] = self.sim.particles[name].ay
            r2_2dot[self._iz, i] = self.sim.particles[name].az
            # planet
            name = self.planet.name
            rp[self._ix, i] = self.sim.particles[name].x
            rp[self._iy, i] = self.sim.particles[name].y
            rp[self._iz, i] = self.sim.particles[name].z
            rp_dot[self._ix, i] = self.sim.particles[name].vx
            rp_dot[self._iy, i] = self.sim.particles[name].vy
            rp_dot[self._iz, i] = self.sim.particles[name].vz
            rp_2dot[self._ix, i] = self.sim.particles[name].ax
            rp_2dot[self._iy, i] = self.sim.particles[name].ay
            rp_2dot[self._iz, i] = self.sim.particles[name].az
        self.r1 = np.append(self.r1, r1, axis=1)
        self.r1_dot = np.append(self.r1_dot, r1_dot, axis=1)
        self.r1_2dot = np.append(self.r1_2dot, r1_2dot, axis=1)
        self.r2 = np.append(self.r2, r2, axis=1)
        self.r2_dot = np.append(self.r2_dot, r2_dot, axis=1)
        self.r2_2dot = np.append(self.r2_2dot, r2_2dot, axis=1)
        self.rp = np.append(self.rp, rp, axis=1)
        self.rp_dot = np.append(self.rp_dot, rp_dot, axis=1)
        self.rp_2dot = np.append(self.rp_2dot, rp_2dot, axis=1)
        self.t = np.append(self.t, times)

    def integrate_orbits(
        self,
        n_orbits: int,
        capture_freq: float = 1.0,
        verbose: int = 1,
        wrapper: Callable = None
    ):
        """
        Integrates simulation according to the orbital period of the planet.

        Parameters
        ----------
        n_orbits : int
            The number of orbits to integrate.
        capture_freq : float, optional
            The frequency of capturing the orbits. Defaults to 1.0.
        verbose : int, optional
            The verbosity level. Defaults to 1.
        wrapper : Callable, optional
            A wrapper function. Defaults to None.

        Returns
        -------
        None
        """
        start_time = 0 if self.t.size == 0 else self.t[-1]
        delta_time_total = self.sim.particles['p'].P*n_orbits
        delta_time_capture = self.sim.particles['p'].P/capture_freq

        times = (start_time + np.arange(start=0,
                 stop=delta_time_total, step=delta_time_capture)[1:])
        self.integrate(times, verbose=verbose, wrapper=wrapper)

    def integrate_to_get_state(
        self,
        step=5,
        max_orbits=1000,
        capture_freq=1
    ):
        """
        Integrates the system the minimum amount of time to
        determine its state.

        Parameters
        ----------
        step : int, optional
            The step size for integration in units of orbits. Defaults to 5.
        max_orbits : int, optional
            The maximum number of orbits to integrate. Defaults to 1000.
        capture_freq : int, optional
            The frequency at which to capture the state in units of 1/orbit. Defaults to 1.

        Raises
        ------
        RuntimeError
            If the total number of orbits exceeds the maximum limit.

        Returns
        -------
        None
        """
        tot_orbits = 0
        while self.state == UNKNOWN:
            wrapper = get_wrapper(
                desc=f'Integrating (max={max_orbits})',
                total=tot_orbits+step,
                start=tot_orbits
            )
            self.integrate_orbits(
                step, verbose=1, wrapper=wrapper, capture_freq=capture_freq)
            tot_orbits += step
            if tot_orbits > max_orbits:
                raise RuntimeError(f'Reached limit of max_orbits={max_orbits}')

    def integrate_to_get_path(
        self,
        step=5,
        max_orbits=1000,
        capture_freq=1
    ):
        """
        Integrate the system just long enough to capture a full path
        of procession. This in nice for creating plots.

        Parameters
        ----------
        step : int, optional
            The step size for integration in units of orbits. Defaults to 5.
        max_orbits : int, optional
            The maximum number of orbits to integrate. Defaults to 1000.
        capture_freq : int, optional
            The frequency at which to capture the state in units of 1/orbit. Defaults to 1.

        Raises
        ------
        RuntimeError
            If the total number of orbits exceeds the maximum limit.
        """
        tot_orbits = 0
        while not self.has_returned:
            wrapper = get_wrapper(
                desc=f'Integrating (max={max_orbits})',
                total=tot_orbits+step,
                start=tot_orbits
            )
            self.integrate_orbits(
                step, verbose=1, wrapper=wrapper, capture_freq=capture_freq)
            tot_orbits += step
            if tot_orbits > max_orbits:
                raise RuntimeError(f'Reached limit of max_orbits={max_orbits}')

    @property
    def has_returned(self) -> bool:
        """
        True if the planet has returned to it's original position.

        Returns
        -------
        bool
            Whether or not the planet has returned to it's original position.

        Warns
        -----
        RuntimeWarning
            If the planet's mass is not zero.
        """
        # if self.planet.mass != 0:
        #     msg = 'A planet with mass > 0 is not guaranteed to return to it\'s original position.'
        #     warnings.warn(msg, RuntimeWarning)
        if len(self.t) < 2:
            return False
        if self.t[-1] < 2 * np.pi * np.sqrt(self.planet.semimajor_axis**3 / params.G / self.binary.mass_binary):
            return False
        else:
            x = self.icosomega
            y = self.isinomega
            dx = x[1]-x[0]
            dy = y[1]-y[0]
            dr = np.sqrt(dx**2 + dy**2)
            r = np.sqrt((x-x[0])**2 + (y-y[0])**2)
            return np.any(r[2:] < dr)

    @property
    def angular_momentum_binary(self) -> np.ndarray:
        """
        The angular momentum of the Binary, computed in the CoM frame of the binary.

        :type: numpy.ndarray, size=(3,N)
        """
        l1 = cross(self.r1, self.r1_dot)*self.binary.mass1
        l2 = cross(self.r2, self.r2_dot)*self.binary.mass2
        return l1 + l2

    @property
    def eccentricity_bin(self) -> np.ndarray:
        """
        The eccentricity vector computed using m1 & m2.

        Notes
        -----
        \\vec{e} = \\frac{\\vec{v} \\times \\vec{h}}{\\mu} - \\frac{\\vec{r}}{r}
        """
        r = self.r2 - self.r1
        v = self.r2_dot - self.r1_dot
        r_mag = np.sqrt(dot(r, r))
        h = cross(r, v)
        mu = params.G * (self.binary.mass1 + self.binary.mass2)
        # let e = a-b
        a = cross(v, h) / mu
        b = r / r_mag
        return a-b

    @property
    def z_hat(self) -> np.ndarray:
        """
        The direction of the angular momentum of the binary.
        """
        l = self.angular_momentum_binary
        l_mag = np.sqrt(dot(l, l))
        return l/l_mag

    @property
    def x_hat(self) -> np.ndarray:
        """
        The direction of the eccentricity vector of the binary,
        computed using values for m2.
        For zero eccentricity, this value is chosen to be [1.,0.,0.]
        """
        threshold = 1e-6
        e = self.eccentricity_bin
        e_mag = np.sqrt(dot(e, e))
        zero_e = e_mag < threshold
        return np.where(~zero_e, e/e_mag, np.array([[1., 0., 0.]]).T)

    @property
    def y_hat(self) -> np.ndarray:
        """
        The y direction in our coordinate system.
        """
        return -cross(self.x_hat, self.z_hat)

    @property
    def specific_angular_momentum(self) -> np.ndarray:
        """
        The angular momentum of the planet per unit mass.

        Notes
        -----
        .. math::
            \\vec{h} = \\vec{r} \\times \\vec{v}
        """
        return cross(self.rp, self.rp_dot)

    @property
    def specific_torque(self) -> np.ndarray:
        """
        The torque per unit mass on the planet.

        Notes
        -----
        .. math::
            \\vec{\\tau} = \\vec{r} \\times \\vec{a}
        """
        return cross(self.rp, self.rp_2dot)

    @property
    def inclination(self) -> np.ndarray:
        """
        The inclination of the planet in radians.

        Notes
        -----
        .. math::
            \\cos{i} = \\frac{h_z}{h}

        where
        .. math::
            \\vec{h} = \\vec{r} \\times \\vec{v}

        is the specific angular momentum.
        """
        h = self.specific_angular_momentum
        h_z = dot(h, self.z_hat)
        h_mag = np.sqrt(dot(h, h))
        return np.arccos(h_z/h_mag)

    @property
    def inclination_dot(self) -> np.ndarray:
        """
        The time derivitive of the inclination.

        Notes
        -----
        Recall
        .. math::
            i = \\arccos{\\frac{h_z}{h}}

        It follows that
        .. math::
            \\frac{di}{dt} = \\frac{-1}{\\sqrt{1 - (\\frac{h_z}{h})^2}}
            \\frac{d}{dt}(\\frac{h_z}{h})

        and
        .. math::
            \\frac{d}{dt}(\\frac{h_z}{h}) = \\frac{\\dot{h_z}h - h_z\\dot{h}}{h^2}

        where
        .. math::
            h = \\sqrt{\\vec{h} \\cdot \\vec{h}}

        .. math::
            \\vec{h} = \\vec{r} \\times \\vec{v}

        and
        .. math::
            h_z = \\vec{h} \\cdot \\hat{z}

        We can see that
        .. math::
            \\dot{h_z} = \\frac{d}{dt} (\\vec{h} \\cdot \\hat{z}) = \\dot{\\vec{h}} \\cdot \\hat{z}

        and
        .. math::
            \\dot{h} = \\frac{d}{dt} (\\sqrt{\\vec{h} \\cdot \\vec{h}})
            = \\frac{1}{2} \\frac{1}{\\sqrt{\\vec{h} \\cdot \\vec{h}}}
            (2\\dot{\\vec{h}} \\cdot \\vec{h})
            = \\frac{\\dot{\\vec{h}} \\cdot \\vec{h}}{\\sqrt{\\vec{h} \\cdot \\vec{h}}}

        where
        .. math::
            \\dot{\\vec{h}} = \\frac{d}{dt} (\\vec{r} \\times \\vec{v})
            = \\dot{\\vec{r}} \\times \\vec{v} + \\vec{r} \\times \\dot{\\vec{v}}
            = \\vec{v} \\times \\vec{v} + \\vec{r} \\times \\vec{a}
            = \\vec{r} \\times \\vec{a} = \\vec{\\tau}

        is the specific torque.

        Working backwards
        .. math::
            \\dot{h} = \\frac{{\\vec{\\tau} \\cdot \\vec{h}} {\\abs{h}}

        .. math::
            \\dot{h_z} = \\vec{\\tau} \\cdot \\hat{z}

        Finally, we evaluate.
        """
        h = self.specific_angular_momentum
        tau = self.specific_torque
        h_z = dot(h, self.z_hat)
        h_mag = np.sqrt(dot(h, h))
        q = h_z/h_mag
        h_z_dot = dot(tau, self.z_hat)
        h_mag_dot = dot(tau, h)/h_mag
        q_dot = (h_z_dot*h_mag - h_z*h_mag_dot)/h_mag**2
        num = -q_dot
        den = np.sqrt(1-q**2)
        indeterminant = (num == 0) & (den == 0)  # choose these values to be 0
        i_dot = np.where(~indeterminant, num/den, 0)
        return i_dot

    @property
    def lon_ascending_node(self) -> np.ndarray:
        """
        The Longitude of the Ascending Node.

        Notes
        -----
        .. math::
            \\Omega = \\texttt{atan2(}h_x, -h_y \\texttt{)}
        """
        h = self.specific_angular_momentum
        h_x = dot(h, self.x_hat)
        h_y = dot(h, self.y_hat)
        return np.unwrap(np.arctan2(h_x, -h_y))

    @property
    def lon_ascending_node_dot(self) -> np.ndarray:
        """
        The time derivative of the longitude of the ascending node.

        In the case of nans, that means that there is a degeneracy
        with argument of pariapsis (and the orbit is coplanar with
        the binary). This should not happen in practice, and if it does,
        reconsider your choice of parameters (e.g. set binary eccentricity to
        1e-6 rather than 0).

        Notes
        -----
        .. math::
            \\Omega = \\texttt{atan2(}h_x, -h_y \\texttt{)}

        and
        .. math::
            \\frac{d\\Omega}{dt} = \\frac{h_y}{h_{x}^2 + h_{y}^2} \\frac{dh_x}{dt}
                + \\frac{h_x}{h_{x}^2 + h_{y}^2} \\frac{-dh_y}{dt}
        """
        h = self.specific_angular_momentum
        tau = self.specific_torque
        h_x = dot(h, self.x_hat)
        h_y = dot(h, self.y_hat)
        h_x_dot = dot(tau, self.x_hat)
        h_y_dot = dot(tau, self.y_hat)
        den = h_x**2 + h_y**2
        num = h_y * h_x_dot - h_x * h_y_dot
        inteterminant = (num == 0) & (den == 0)  # choose these values to be 0
        return np.where(inteterminant, 0, num/den)

    @property
    def icosomega(self) -> np.ndarray:
        """
        The inclination times the cosine of the longitude of the ascending node.

        Notes
        -----
        .. math::
            i \\cos{\\Omega}
        """
        return self.inclination * np.cos(self.lon_ascending_node)

    @property
    def isinomega(self) -> np.ndarray:
        """
        The inclination times the sine of the longitude of the ascending node.

        Notes
        -----
        .. math::
            i \\sin{\\Omega}
        """
        return self.inclination * np.sin(self.lon_ascending_node)

    @property
    def icosomega_dot(self) -> np.ndarray:
        """
        The time derivitive of :math:`i\\cos{\\Omega}`

        Notes
        -----
        let
        .. math::
            f = i \\cos{\\Omega}

        then
        .. math::
            \\frac{df}{dt} = \\dot{i} \\cos{\\Omega} + i \\frac{d}{dt}(\\cos{\\Omega})
            = \\dot{i} \\cos{\\Omega} - i \\sin{\\Omega} \\dot{Omega}
        """
        i_dot = self.inclination_dot
        i = self.inclination
        omega = self.lon_ascending_node
        omega_dot = self.lon_ascending_node_dot
        f_dot = i_dot * np.cos(omega) - i * np.sin(omega) * omega_dot
        return f_dot

    @property
    def isinomega_dot(self) -> np.ndarray:
        """
        The time derivitive of :math:`i\\sin{\\Omega}`

        Notes
        -----
        let
        .. math::
            f = i \\sin{\\Omega}

        then
        .. math::
            \\frac{df}{dt} = \\dot{i} \\sin{\\Omega} + i \\frac{d}{dt}(\\sin{\\Omega})
            = \\dot{i} \\sin{\\Omega} + i \\cos{\\Omega} \\dot{Omega}
        """
        i_dot = self.inclination_dot
        i = self.inclination
        omega = self.lon_ascending_node
        omega_dot = self.lon_ascending_node_dot
        f_dot = i_dot * np.sin(omega) + i * np.cos(omega) * omega_dot
        return f_dot

    @property
    def _lon_ascending_node_diff(self) -> np.ndarray:
        """
        The numerical change in the longitude of the ascending node,
        excluding shifts by 2 pi.
        """
        omega = self.lon_ascending_node
                
        _, delta_omega = central_difference(self.t, omega, 2)
        two_pi_shift = np.isclose(np.abs(delta_omega), 2*np.pi, atol=0.3)
        delta_omega = delta_omega[~two_pi_shift]
        return delta_omega

    @property
    def _lon_ascending_node_ndir(self) -> Tuple[int, int]:
        """
        The number of positive and negative shifts in the longitude of the ascending node.
        This will detect changes of direction.
        """
        omega_diff = self._lon_ascending_node_diff
        positive = omega_diff > 0
        negative = omega_diff < 0
        zero = omega_diff == 0
        two_positive_in_a_row = (positive[1:] & positive[:-1])
        two_negative_in_a_row = (negative[1:] & negative[:-1])
        npos = np.sum(two_positive_in_a_row)
        nneg = np.sum(two_negative_in_a_row)
        return npos, nneg

    @property
    def _crossed_x_axis(self) -> bool:
        """
        True if the orbit crossed the x axis in (lon ascending node, inclination) space.
        """
        y = self.isinomega
        return (np.sum(y > 0) > 0) & (np.sum(y < 0) > 0)

    @property
    def _turned_around(self) -> bool:
        """
        True if the orbit has turned around.
        """
        npos, nneg = self._lon_ascending_node_ndir
        if npos == 0:
            return False
        elif nneg == 0:
            return False
        else:
            return True

    @property
    def _state_lowj(self) -> str:
        """
        The current state of the orbit.

        Returns
        -------
        str
            The state of the orbit.
        """
        has_turned = self._turned_around
        has_crossed = self._crossed_x_axis
        if not has_turned and not has_crossed:  # we don't yet know anything
            return UNKNOWN
        elif has_turned:
            return LIBRATING
        else:
            npos, nneg = self._lon_ascending_node_ndir
            if npos > 0:
                return RETROGRADE
            elif nneg > 0:
                return PROGRADE
            else:
                return UNKNOWN
    @property
    def _state(self)->str:
        """
        State determination for high j orbits.
        """
        if self.has_returned:
            if self._crossed_x_axis:
                domega = self._lon_ascending_node_diff
                domega_mean = np.mean(domega)
                if domega_mean > 0:
                    return RETROGRADE
                else:
                    return PROGRADE
            else:
                return LIBRATING
        else:
            return UNKNOWN
    
    @property
    def state(self)->str:
        """
        Get the current state of the orbit.
        
        Returns
        -------
        str
            The state of the orbit.
        """
        jcrit = get_j_crit(self.binary.eccentricity_binary)
        j = get_j(
            mb=self.binary.mass_binary,
            fb=self.binary.mass_fraction,
            ab=self.binary.semimajor_axis_binary,
            eb=self.binary.eccentricity_binary,
            mp=self.planet.mass,
            ap=self.planet.semimajor_axis
        )
        if j < 0.1*jcrit:
            return self._state_lowj
        else:
            return self._state

    @property
    def _area(self) -> float:
        """
        The area of the parameter space enclosed by the orbit.
        This value is uncorrected and the direction must be taken
        into account.
        """
        if not self.has_returned:
            raise RuntimeError('System has not yet returned.')
        else:
            return integrate_r_and_theta(
                *cast_r_and_theta(
                    self.inclination,
                    self.lon_ascending_node
                )
            )

    @property
    def normalized_area(self) -> float:
        """
        The fraction of the parameter space enclosed by the orbit.

        Returns
        -------
        float
            The fraction of the parameter space enclosed by the orbit. Range is [0,1].
        """
        area = self._area
        if self.state == RETROGRADE:
            area = 4*np.pi + area
        return np.abs(area/(4*np.pi))
