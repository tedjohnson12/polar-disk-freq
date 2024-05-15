"""
Database module to store results of MC simulations.
"""
from pathlib import Path
import sqlite3

TABLE_NAME = 'mc'
TABLE_COLS = [
    'mass_binary',
    'mass_fraction',
    'semimajor_axis_binary',
    'eccentricity_binary',
    'mass_planet',
    'semimajor_axis_planet',
    'true_anomaly_planet',
    'eccentricity_planet',
    'arg_pariapsis_planet',
    'inclination',
    'lon_ascending_node',
    'has_gr',
    'state'
]

def fmt(val: float)->str:
    if isinstance(val, float):
        return f'{val:.6e}'
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return f'\'{val}\''
    return str(val)

def connect(path: Path)->sqlite3.Connection:
    """
    Connect to the database.

    Parameters
    ----------
    path : Path
        The path to the database.
    """
    if path is None:
        return None
    return sqlite3.connect(path)

def is_empty(conn: sqlite3.Connection):
    """
    Check if the database is empty.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    """
    cur = conn.cursor()
    cur.execute('SELECT name FROM sqlite_master')
    return cur.fetchone() is None

def setup(
    conn: sqlite3.Connection,
    overwrite=False
):
    """
    Set up the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    overwrite : bool, optional
        Whether or not to overwrite the table if it already exists.
    """
    cur = conn.cursor()
    create_str = 'CREATE TABLE' if overwrite else 'CREATE TABLE IF NOT EXISTS'
    cur.execute(
        f'{create_str} {TABLE_NAME} ({",".join(TABLE_COLS)})'
    )

def insert(
    conn: sqlite3.Connection,
    mass_binary: float,
    mass_fraction: float,
    semimajor_axis_binary: float,
    eccentricity_binary: float,
    mass_planet: float,
    semimajor_axis_planet: float,
    true_anomaly_planet: float,
    eccentricity_planet: float,
    arg_pariapsis_planet: float,
    inclination: float,
    lon_ascending_node: float,
    has_gr: bool,
    state: str,
    commit=True
):
    """
    Insert a row into the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    mass_binary : float
        The mass of the binary.
    mass_fraction : float
        The mass fraction of the binary.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.
    mass_planet : float
        The mass of the planet.
    semimajor_axis_planet : float
        The semimajor axis of the planet.
    true_anomaly_planet : float
        The true anomaly of the planet.
    eccentricity_planet : float
        The eccentricity of the planet.
    arg_pariapsis_planet : float
        The argument of the periapsis of the planet.
    inclination : float
        The inclination of the orbit.
    lon_ascending_node : float
        The longitude of the ascending node.
    has_gr : bool
        Whether or not General Relativity is inclid
    state : str
        The state of the orbit.
    commit : bool, optional
        Whether or not to commit the changes, by default True.
    """
    cur = conn.cursor()
    val_str = f'{fmt(mass_binary)},{fmt(mass_fraction)}' + \
        f',{fmt(semimajor_axis_binary)},{fmt(eccentricity_binary)}' + \
        f',{fmt(mass_planet)},{fmt(semimajor_axis_planet)}' + \
        f',{fmt(true_anomaly_planet)},{fmt(eccentricity_planet)}' + \
        f',{fmt(arg_pariapsis_planet)},{fmt(inclination)},{fmt(lon_ascending_node)},{fmt(has_gr)},{fmt(state)}'
    cur.execute(
        f'INSERT INTO {TABLE_NAME} VALUES ({val_str})'
    )
    if commit:
        conn.commit()

def get_var(
    conn: sqlite3.Connection,
    variable:str,
    mass_binary: float=None,
    mass_fraction: float=None,
    semimajor_axis_binary: float=None,
    eccentricity_binary: float=None,
    mass_planet: float=None,
    semimajor_axis_planet: float=None,
    true_anomaly_planet: float=None,
    eccentricity_planet: float=None,
    arg_pariapsis_planet: float=None,
    inclination: float=None,
    lon_ascending_node: float=None,
    has_gr: bool=None
):
    """
    Get the state of the orbit.

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    mass_binary : float
        The mass of the binary.
    mass_fraction : float
        The mass fraction of the binary.
    semimajor_axis_binary : float
        The semimajor axis of the binary.
    eccentricity_binary : float
        The eccentricity of the binary.
    mass_planet : float
        The mass of the planet.
    semimajor_axis_planet : float
        The semimajor axis of the planet.
    true_anomaly_planet : float
        The true anomaly of the planet.
    eccentricity_planet : float
        The eccentricity of the planet.
    arg_pariapsis_planet : float
        The argument of the periapsis of the planet.
    inclination : float
        The inclination of the orbit.
    lon_ascending_node : float
        The longitude of the ascending node.
    has_gr : bool
        Whether or not General Relativity is included
    """
    cur = conn.cursor()
    values = (
        ('mass_binary', mass_binary),
        ('mass_fraction', mass_fraction),
        ('semimajor_axis_binary', semimajor_axis_binary),
        ('eccentricity_binary', eccentricity_binary),
        ('mass_planet', mass_planet),
        ('semimajor_axis_planet', semimajor_axis_planet),
        ('true_anomaly_planet', true_anomaly_planet),
        ('eccentricity_planet', eccentricity_planet),
        ('arg_pariapsis_planet', arg_pariapsis_planet),
        ('inclination', inclination),
        ('lon_ascending_node', lon_ascending_node),
        ('has_gr', has_gr)
    )
    val_str = ' AND '.join(f'{k}={fmt(v)}' for k, v in values if v is not None)
    res = cur.execute(
        f'SELECT {variable} FROM {TABLE_NAME} WHERE {val_str}'
    )
    return res.fetchall()