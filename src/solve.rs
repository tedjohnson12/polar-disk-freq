// Solve Martin & Lubow eq (7-10) using RK4


use ndarray::Array1;
use pyo3::pyfunction;
use std::time::{SystemTime,Duration};
use std::collections::HashSet;
use std::f64::consts::PI;

/// This is our set of ODEs
fn derivatives(
    _tau: f64,
    lx: f64,
    ly: f64,
    lz: f64,
    eb: f64,
    gamma: f64,
) -> (f64, f64, f64, f64) {

    let one_minus_eb2 = 1.0 - eb.powf(2.0);

    let dlxdtau: f64 = one_minus_eb2*ly*lz + gamma * one_minus_eb2.sqrt() * ly * (2. - 5.0*lx.powf(2.0));
    let dlydtau: f64 = -(1.0+4.0*eb.powf(2.0))*lx*lz - gamma*lx/one_minus_eb2.sqrt()* ( one_minus_eb2*(2.0-5.0*lx.powf(2.0)) + 5.*eb.powf(2.)*lz.powf(2.) );
    let dlzdtau: f64 = 5.*eb*lx*ly + 5.*gamma*eb.powf(2.)/one_minus_eb2.sqrt()*lx*ly*lz;
    let debdtau: f64 = 5.*gamma*eb*one_minus_eb2.sqrt()*lx*ly;

    (dlxdtau, dlydtau, dlzdtau, debdtau)

}

/// gamma is a constant of motion. See Martin & Lubow (2019) eq (11,12)
#[pyfunction]
pub fn get_gamma(eb:f64,j:f64)-> f64 {
    (1.0 - eb.powf(2.)).sqrt() * j
}

/// Notice we rotate by pi/2 in Omega
#[pyfunction]
pub fn init_xyz(
    i: f64,
    omega: f64,
) -> (f64, f64, f64) {
    (i.sin() * (omega-PI/2.).cos(), i.sin() * (omega-PI/2.).sin(), i.cos())
}

/// Fourth order Runga-Kutta
fn rk4(
    tau: f64,
    dtau: f64,
    lx: f64,
    ly: f64,
    lz: f64,
    eb: f64,
    gamma: f64,
 ) -> (f64, f64, f64, f64) {

    let (dlx, dly, dlz, deb) = derivatives(tau, lx, ly, lz, eb, gamma);
    
    let k1_lx = dtau * dlx;
    let k1_ly = dtau * dly;
    let k1_lz = dtau * dlz;
    let k1_eb = dtau * deb;

    let (dlx, dly, dlz, deb) = derivatives(tau+dtau/2.0,lx+0.5*k1_lx,ly+0.5*k1_ly,lz+0.5*k1_lz,eb+0.5*k1_eb,gamma);

    let k2_lx = dtau * dlx;
    let k2_ly = dtau * dly;
    let k2_lz = dtau * dlz;
    let k2_eb = dtau * deb;

    let (dlx, dly, dlz, deb) = derivatives(tau+dtau/2.0,lx+0.5*k2_lx,ly+0.5*k2_ly,lz+0.5*k2_lz,eb+0.5*k2_eb,gamma);

    let k3_lx = dtau * dlx;
    let k3_ly = dtau * dly;
    let k3_lz = dtau * dlz;
    let k3_eb = dtau * deb;

    let (dlx, dly, dlz, deb) = derivatives(tau+dtau,lx+k3_lx,ly+k3_ly,lz+k3_lz,eb+k3_eb,gamma);

    let k4_lx = dtau * dlx;
    let k4_ly = dtau * dly;
    let k4_lz = dtau * dlz;
    let k4_eb = dtau * deb;

    let dlx = 1.0/6.0 * (k1_lx + 2.0*k2_lx + 2.0*k3_lx + k4_lx);
    let dly = 1.0/6.0 * (k1_ly + 2.0*k2_ly + 2.0*k3_ly + k4_ly);
    let dlz = 1.0/6.0 * (k1_lz + 2.0*k2_lz + 2.0*k3_lz + k4_lz);
    let deb = 1.0/6.0 * (k1_eb + 2.0*k2_eb + 2.0*k3_eb + k4_eb);

    (lx + dlx, ly + dly, lz + dlz, eb + deb)

}

/// States. We don't distinguish between libration and crescent orbits here.
/// It is not necessary and also difficult.
#[derive(PartialEq,Debug)]
pub enum State {
    Prograde,
    Retrograde,
    Librating,
    Unknown
}

/// mutual inclination
fn get_i(
    lx: f64,
    ly: f64,
    lz: f64,
) -> f64 {

    (lz / (lx.powf(2.)+ly.powf(2.)+lz.powf(2.)).sqrt()).acos()

}

/// Longitude of ascending node
fn get_omega(
    lx: f64,
    ly: f64,
) -> f64 {
    lx.atan2(-ly)
}

/// Where are we in i-Omega space?
#[derive(PartialEq)]
enum Quadrant {
    O,
    X,
    Y,
    I,
    II,
    III,
    IV
}

/// How do we move in i-Omega space?
#[derive(PartialEq,Hash,Eq)]
enum QuadrantTransition {
    I_II,
    II_I,
    II_III,
    III_II,
    III_IV,
    IV_III,
    IV_I,
    I_IV
}

impl QuadrantTransition {
    fn new(q1: &Quadrant, q2: &Quadrant) -> Result<QuadrantTransition, &'static str> {
        match (q1, q2) {
            (Quadrant::I, Quadrant::II) => Result::Ok(QuadrantTransition::I_II),
            (Quadrant::II, Quadrant::I) => Result::Ok(QuadrantTransition::II_I),
            (Quadrant::II, Quadrant::III) => Result::Ok(QuadrantTransition::II_III),
            (Quadrant::III, Quadrant::II) => Result::Ok(QuadrantTransition::III_II),
            (Quadrant::III, Quadrant::IV) => Result::Ok(QuadrantTransition::III_IV),
            (Quadrant::IV, Quadrant::III) => Result::Ok(QuadrantTransition::IV_III),
            (Quadrant::IV, Quadrant::I) => Result::Ok(QuadrantTransition::IV_I),
            (Quadrant::I, Quadrant::IV) => Result::Ok(QuadrantTransition::I_IV),
            _ => Result::Err("Invalid quadrant transition"),
        }
    }
    /// Retrograde orbits precess in the positive direction in i-Omega space
    fn positive(&self) -> bool {
        match self {
            QuadrantTransition::I_II => true,
            QuadrantTransition::II_I => false,
            QuadrantTransition::II_III => true,
            QuadrantTransition::III_II => false,
            QuadrantTransition::III_IV => true,
            QuadrantTransition::IV_III => false,
            QuadrantTransition::IV_I => true,
            QuadrantTransition::I_IV => false,
        }
    }
    /// Librating orbits do not cross the x-axis
    fn across_x(&self) -> bool {
        match self {
            QuadrantTransition::I_II => false,
            QuadrantTransition::II_I => false,
            QuadrantTransition::II_III => true,
            QuadrantTransition::III_II => true,
            QuadrantTransition::III_IV => false,
            QuadrantTransition::IV_III => false,
            QuadrantTransition::IV_I => true,
            QuadrantTransition::I_IV => true,
        }
    }
}

/// Mapping from x-y to quadrant label
fn get_quadrant(
    x: f64,
    y: f64,
) -> Quadrant {
    if x==0.0 && y==0.0 {
        Quadrant::O
    }
    else if x==0.0 {
        Quadrant::Y
    }
    else if y==0.0 {
        Quadrant::X
    }
    else if x>0.0 && y>0.0 {
        Quadrant::I
    }
    else if x<0.0 && y>0.0 {
        Quadrant::II
    }
    else if x<0.0 && y<0.0 {
        Quadrant::III
    }
    else if x>0.0 && y<0.0 {
        Quadrant::IV
    }
    else {
        panic!("Invalid quadrant")
    }
}

/// Map a set of quadrant transitions to a state.
fn eval_history(
    hist: &HashSet<QuadrantTransition>
) -> Result<State, &'static str> {
    let mut crossed_x = false;
    let mut has_positive = false;
    let mut has_negative = false;
    for tr in hist {
        if tr.across_x() {
            crossed_x = true;
        }
        if tr.positive() {
            has_positive = true;
        }
        else {
            has_negative = true;
        }
        if has_negative && has_positive && crossed_x {
            return Result::Err("Cannot have all three");
        }
        else if has_negative && has_positive {
            return Result::Ok(State::Librating);
        }
        else if crossed_x && has_negative {
            return Result::Ok(State::Retrograde);
        }
        else if crossed_x && has_positive {
            return Result::Ok(State::Prograde);
        }
    }
    Result::Ok(State::Unknown)
}

/// Our results in rust
#[derive(Debug)]
pub struct SimResult {
    pub tau: Array1<f64>,
    pub lx: Array1<f64>,
    pub ly: Array1<f64>,
    pub lz: Array1<f64>,
    pub eb: Array1<f64>,
    pub state: State
}

/// integrate in rk4
pub fn integrate(
    tau_init: f64,
    dtau: f64,
    lx_init: f64,
    ly_init: f64,
    lz_init: f64,
    eb_init: f64,
    gamma: f64,
    walltime: f64,
) -> SimResult {
    let start = SystemTime::now();
    let end = start + Duration::from_secs_f64(walltime);
    let mut tau = tau_init;
    let mut lx = lx_init;
    let mut ly = ly_init;
    let mut lz = lz_init;
    let mut eb = eb_init;

    let mut tau_arr = vec![tau_init];
    let mut lx_arr = vec![lx_init];
    let mut ly_arr = vec![ly_init];
    let mut lz_arr = vec![lz_init];
    let mut eb_arr = vec![eb_init];

    let mut i = get_i(lx, ly, lz);
    let mut omega = get_omega(lx, ly);
    
    let mut x = i * omega.cos();
    let mut y = i * omega.sin();
    let mut quad = get_quadrant(x, y);

    let mut hist = HashSet::<QuadrantTransition>::new();
    let mut current_state = State::Unknown;

    

    while SystemTime::now() < end {
        (lx, ly, lz, eb) = rk4(tau, dtau, lx, ly, lz, eb, gamma);
        tau += dtau;
        tau_arr.push(tau);
        lx_arr.push(lx);
        ly_arr.push(ly);
        lz_arr.push(lz);
        eb_arr.push(eb);
        
        i = get_i(lx, ly, lz);
        omega = get_omega(lx, ly);
        x = i * omega.cos();
        y = i * omega.sin();
        let next_quad = get_quadrant(x, y);
        if next_quad != quad {
            let transition = QuadrantTransition::new(&quad, &next_quad);
            if matches!(transition,Result::Ok(_)) {
                hist.insert(transition.unwrap());
            }
        }
        current_state = eval_history(&hist).unwrap();
        if !matches!(current_state, State::Unknown) {
            break
        }
        quad = next_quad;


    }

    SimResult {
        tau: Array1::from_vec(tau_arr),
        lx: Array1::from_vec(lx_arr),
        ly: Array1::from_vec(ly_arr),
        lz: Array1::from_vec(lz_arr),
        eb: Array1::from_vec(eb_arr),
        state: current_state
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let eb = 0.4;
        let j0 = 0.1;
        let gamma = get_gamma(eb, j0);
        let (lx,ly,lz) = init_xyz(0.4*PI,PI/2.0);
        println!("{}",get_i(lx, ly, lz));
        println!("{}",get_omega(lx, ly));
        let start_time = SystemTime::now();
        let res = integrate(0.0, 0.1, lx, ly, lz, eb, gamma, 10.0);
        let dtime = start_time.elapsed().unwrap().as_secs_f64();
        println!("{:?}", res);
        println!("Time integrating: {:?}", dtime);
        println!("Time for 1000: {:?}", dtime*1000.0);
        assert_eq!(res.state, State::Librating);
    }

}