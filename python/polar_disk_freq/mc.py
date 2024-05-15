"""
Monte Carlo Simulation module
"""
from typing import List
from pathlib import Path
import numpy as np
import rebound
from scipy.stats import bootstrap
from tqdm.auto import trange
import warnings

from . import params
from .system import System, UNKNOWN
from . import db

def inclination_transform(u:float)->float:
    """
    Transform :math:`u \\in [0,1)` to :math:`i \\in [0,\\pi)`
    
    Parameters
    ----------
    u : float
        The uniform random variable
    
    Returns
    -------
    float
        The transformed random variable
    """
    return np.arcsin(2*u - 1) + np.pi/2

class Sampler:
    """
    Monte Carlo sampler
    """
    _integration_orbit_step = 5
    _integration_max_orbits = 10000
    _integration_capture_freq = 1
    def __init__(
        self,
        mass_binary: float,
        mass_fraction: float,
        semimajor_axis_binary: float,
        eccentricity_binary: float,
        mass_planet: float,
        semimajor_axis_planet: float,
        true_anomaly_planet: float,
        eccentricity_planet: float = 0,
        arg_pariapsis_planet: float = 0,
        gr: bool = False,
        rng: np.random.Generator = None,
        db_path: Path = None
    ):
        self._binary = params.Binary(
            mass_binary=mass_binary,
            mass_fraction=mass_fraction,
            semimajor_axis_binary=semimajor_axis_binary,
            eccentricity_binary=eccentricity_binary
        )
        self._mass_planet = mass_planet
        self._semimajor_axis_planet = semimajor_axis_planet
        self._true_anomaly_planet = true_anomaly_planet
        self._eccentricity_planet = eccentricity_planet
        self._arg_pariapsis_planet = arg_pariapsis_planet
        self._gr = gr
        self._db_path = db_path
        self.conn = db.connect(self._db_path)
        if self.conn is not None:
            if db.is_empty(self.conn):
                db.setup(self.conn, overwrite=False)
        self.inclinations: List[float] = self._init_var('inclination')
        self.lon_ascending_nodes: List[float] = self._init_var('lon_ascending_node')
        self.states: List[str] = self._init_var('state')
        if db_path is not None and rng is not None:
            msg = 'Cannot provide a seed and a database simultaneously. '
            msg += 'This would run the risk if repeating the same entries each time the database is added to. '
            msg += 'Entries to the database require fresh entropy, so please set `rng=None`.'
            raise ValueError(msg)
        self.rng = np.random.default_rng() if rng is None else rng
    def _init_var(self,var:str):
        if self.conn is None:
            return []
        res = db.get_var(
            conn=self.conn,
            variable=var,
            mass_binary=self._binary.mass_binary,
            mass_fraction=self._binary.mass_fraction,
            semimajor_axis_binary=self._binary.semimajor_axis_binary,
            eccentricity_binary=self._binary.eccentricity_binary,
            mass_planet=self._mass_planet,
            semimajor_axis_planet=self._semimajor_axis_planet,
            true_anomaly_planet=self._true_anomaly_planet,
            eccentricity_planet=self._eccentricity_planet,
            arg_pariapsis_planet=self._arg_pariapsis_planet,
            has_gr=self._gr
        )
        return [r[0] for r in res]
    
    def _planet(
        self,
        inclination: float,
        lon_ascending_node: float
        ) -> params.Planet:
        """
        Generate a Planet parameter set given an inclination.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node

        Returns
        -------
        params.Planet
            The planet parameters.
        """
        return params.Planet(
            mass=self._mass_planet,
            semimajor_axis=self._semimajor_axis_planet,
            inclination=inclination,
            lon_ascending_node=lon_ascending_node,
            true_anomaly=self._true_anomaly_planet,
            eccentricity=self._eccentricity_planet,
            arg_pariapsis=self._arg_pariapsis_planet
        )
    def get_system(
        self,
        inclination: float,
        lon_ascending_node: float,
        sim: rebound.Simulation = None
        ) -> System:
        """
        Generate a system given an inclination.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node
        sim : rebound.Simulation, optional
            The simulation to add the particles to, by default None
        """
        return System(
            binary=self._binary,
            planet=self._planet(inclination=inclination, lon_ascending_node=lon_ascending_node),
            sim=sim,
            gr=self._gr
        )
    def get_simulation_state(
        self,
        inclination: float,
        lon_ascending_node: float
    ):
        """
        Get the behavior of the system given an inclination.

        This function will create a system and integrate it just long
        enough to see it's long-term behavior.

        Parameters
        ----------
        inclination : float
            The initial inclination
        lon_ascending_node : float
            The initial longitude of the ascending node

        Returns
        -------
        str
            The final state of the system.
        """
        sys = self.get_system(inclination=inclination, lon_ascending_node=lon_ascending_node)
        try:
            sys.integrate_to_get_state(
                step=self._integration_orbit_step,
                max_orbits=self._integration_max_orbits,
                capture_freq=self._integration_capture_freq
            )
        except RuntimeError:
            msg = f'Gave up on integration for i={inclination}, l={lon_ascending_node}'
            warnings.warn(msg, RuntimeWarning)
            return UNKNOWN
        return sys.state
    
    def get_next(self):
        
        next_lon_ascending_node = self.rng.random()*2*np.pi
        x = self.rng.random()
        next_inclination = inclination_transform(x)
        
        state = self.get_simulation_state(
            inclination=next_inclination,
            lon_ascending_node=next_lon_ascending_node
        )
        return next_inclination, next_lon_ascending_node, state
    
    @property
    def n_sampled(self):
        return len(self.states)

    def get_frac(self,state):
        return self.states.count(state)/self.n_sampled
    
    def bootstrap(self,state:str,confidence_level:float=0.95):
        def _statistic(arr:List[str]):
            unique, counts = np.unique(arr, return_counts=True)
            n_tot = len(arr)
            n_state = dict(zip(unique, counts)).get(state,0)
            n_unknown = dict(zip(unique, counts)).get(UNKNOWN,0)
            if state == UNKNOWN:
                return n_unknown/n_tot
            else:
                return (n_state + np.random.random_integers(0,n_unknown))/n_tot
        return bootstrap([self.states], _statistic, confidence_level=confidence_level)
        
    def _db_insert(self,inclination:float,lon_ascending_node:float,state:str,commit:bool=False):
        """
        Insert the data into the database.
        """
        db.insert(
            conn=self.conn,
            mass_binary=self._binary.mass_binary,
            mass_fraction=self._binary.mass_fraction,
            semimajor_axis_binary=self._binary.semimajor_axis_binary,
            eccentricity_binary=self._binary.eccentricity_binary,
            mass_planet=self._mass_planet,
            semimajor_axis_planet=self._semimajor_axis_planet,
            true_anomaly_planet=self._true_anomaly_planet,
            eccentricity_planet=self._eccentricity_planet,
            arg_pariapsis_planet=self._arg_pariapsis_planet,
            inclination=inclination,
            lon_ascending_node=lon_ascending_node,
            has_gr=self._gr,
            state=state,
            commit=commit
        )
    
    def sim_n_samples(self,N:int):
        for _ in trange(N, desc='Sampling', unit='samples'):
            next_inclination, next_lon_ascending_node, state = self.get_next()
            self.inclinations.append(next_inclination)
            self.lon_ascending_nodes.append(next_lon_ascending_node)
            self.states.append(state)
            if self.conn is not None:
                self._db_insert(inclination=next_inclination,lon_ascending_node=next_lon_ascending_node,state=state)
        if self.conn is not None:
            self.conn.commit()
    
    def get_confidence_interval_width(self,state:str,confidence_level:float=0.95):
        if len(self.states) < 2:
            return np.inf
        res = self.bootstrap(state,confidence_level=confidence_level)
        if np.isnan(res.confidence_interval[0]) or np.isnan(res.confidence_interval[1]):
            return np.inf
        return res.confidence_interval[1] - res.confidence_interval[0]
    
    def sim_until_precision(self,precision:float,batch_size:int=100,max_samples = 1000,confidence_level:float=0.95):
        interval_width = self.get_confidence_interval_width('l',confidence_level=confidence_level)
        print(f'Starting with {self.n_sampled} samples, the confidence interval width is {interval_width:.3f}')
        while self.n_sampled < max_samples and interval_width > precision:
            self.sim_n_samples(batch_size)
            interval_width = self.get_confidence_interval_width('l',confidence_level=0.95)
            print(f'After {self.n_sampled} samples, the confidence interval width is {interval_width:.3f}')
        
        
    