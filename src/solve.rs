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
    let dlzdtau: f64 = 5.*eb.powf(2.)*lx*ly + 5.*gamma*eb.powf(2.)/one_minus_eb2.sqrt()*lx*ly*lz;
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


enum RKResult {
    Ok((f64, f64, f64, f64, f64)),
    Repeat(f64),
    Err(&'static str),
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
    epsilon: f64
 ) -> RKResult {

    let a: f64 = 0.0;
    let (dlx, dly, dlz, deb) = derivatives(tau+a*dtau, lx, ly, lz, eb, gamma);
    let k1_lx = dtau * dlx;
    let k1_ly = dtau * dly;
    let k1_lz = dtau * dlz;
    let k1_eb = dtau * deb;

    let a: f64 = 1.0/4.0;
    let b21: f64 = 1.0/4.0;
    let (dlx, dly, dlz, deb) = derivatives(
        tau+a*dtau,
        lx+b21*k1_lx,
        ly+b21*k1_ly,
        lz+b21*k1_lz,
        eb+b21*k1_eb,
        gamma
    );
    let k2_lx = dtau * dlx;
    let k2_ly = dtau * dly;
    let k2_lz = dtau * dlz;
    let k2_eb = dtau * deb;

    let a: f64 = 3.0/8.0;
    let b31: f64 = 3.0/32.0;
    let b32: f64 = 9.0/32.0;
    let (dlx, dly, dlz, deb) = derivatives(
        tau+a*dtau,
        lx+b31*k1_lx + b32*k2_lx,
        ly+b31*k1_ly + b32*k2_ly,
        lz+b31*k1_lz + b32*k2_lz,
        eb+b31*k1_eb + b32*k2_eb,
        gamma
    );
    let k3_lx = dtau * dlx;
    let k3_ly = dtau * dly;
    let k3_lz = dtau * dlz;
    let k3_eb = dtau * deb;

    let a: f64 = 12.0/13.0;
    let b41: f64 = 1932.0/2197.0;
    let b42: f64 = -7200.0/2197.0;
    let b43: f64 = 7296.0/2197.0;
    let (dlx, dly, dlz, deb) = derivatives(
        tau+a*dtau,
        lx+b41*k1_lx + b42*k2_lx + b43*k3_lx,
        ly+b41*k1_ly + b42*k2_ly + b43*k3_ly,
        lz+b41*k1_lz + b42*k2_lz + b43*k3_lz,
        eb+b41*k1_eb + b42*k2_eb + b43*k3_eb,
        gamma
    );
    let k4_lx = dtau * dlx;
    let k4_ly = dtau * dly;
    let k4_lz = dtau * dlz;
    let k4_eb = dtau * deb;

    let a: f64 = 1.0;
    let b51: f64 = 439.0/216.0;
    let b52: f64 = -8.0;
    let b53: f64 = 3680.0/513.0;
    let b54: f64 = -845.0/4104.0;
    let (dlx, dly, dlz, deb) = derivatives(
        tau+a*dtau,
        lx+b51*k1_lx + b52*k2_lx + b53*k3_lx + b54*k4_lx,
        ly+b51*k1_ly + b52*k2_ly + b53*k3_ly + b54*k4_ly,
        lz+b51*k1_lz + b52*k2_lz + b53*k3_lz + b54*k4_lz,
        eb+b51*k1_eb + b52*k2_eb + b53*k3_eb + b54*k4_eb,
        gamma
    );
    let k5_lx = dtau * dlx;
    let k5_ly = dtau * dly;
    let k5_lz = dtau * dlz;
    let k5_eb = dtau * deb;

    let a: f64 = 1.0/2.0;
    let b61: f64 = -8.0/27.0;
    let b62: f64 = 2.0;
    let b63: f64 = -3544.0/2565.0;
    let b64: f64 = 1859.0/4104.0;
    let b65: f64 = -11.0/40.0;
    let (dlx, dly, dlz, deb) = derivatives(
        tau+a*dtau,
        lx+b61*k1_lx + b62*k2_lx + b63*k3_lx + b64*k4_lx + b65*k5_lx,
        ly+b61*k1_ly + b62*k2_ly + b63*k3_ly + b64*k4_ly + b65*k5_ly,
        lz+b61*k1_lz + b62*k2_lz + b63*k3_lz + b64*k4_lz + b65*k5_lz,
        eb+b61*k1_eb + b62*k2_eb + b63*k3_eb + b64*k4_eb + b65*k5_eb,
        gamma
    );
    let k6_lx = dtau * dlx;
    let k6_ly = dtau * dly;
    let k6_lz = dtau * dlz;
    let k6_eb = dtau * deb;

    let c1 = 25.0/216.0;
    let c2 = 0.0;
    let c3 = 1408.0/2565.0;
    let c4 = 2197.0/4104.0;
    let c5 = -1.0/5.0;
    
    // let ch1 = 16.0/135.0;
    // let ch2 = 0.0;
    // let ch3 = 6656.0/12825.0;
    // let ch4 = 28561.0/56430.0;
    // let ch5 = -9.0/50.0;
    // let ch6 = 2.0/55.0;

    let ct1 = -1.0/360.0;
    let ct2 = 0.0;
    let ct3 = 128.0/4275.0;
    let ct4 = 2197.0/75240.0;
    let ct5 = -1.0/50.0;
    let ct6 = -2.0/55.0;

    let dlx5 = c1*k1_lx + c2*k2_lx + c3*k3_lx + c4*k4_lx + c5*k5_lx;
    let dly5 = c1*k1_ly + c2*k2_ly + c3*k3_ly + c4*k4_ly + c5*k5_ly;
    let dlz5 = c1*k1_lz + c2*k2_lz + c3*k3_lz + c4*k4_lz + c5*k5_lz;
    let deb5 = c1*k1_eb + c2*k2_eb + c3*k3_eb + c4*k4_eb + c5*k5_eb;

    // let dlx6 = ch1*k1_lx + ch2*k2_lx + ch3*k3_lx + ch4*k4_lx + ch5*k5_lx + ch6*k6_lx;
    // let dly6 = ch1*k1_ly + ch2*k2_ly + ch3*k3_ly + ch4*k4_ly + ch5*k5_ly + ch6*k6_ly;
    // let dlz6 = ch1*k1_lz + ch2*k2_lz + ch3*k3_lz + ch4*k4_lz + ch5*k5_lz + ch6*k6_lz;
    // let deb6 = ch1*k1_eb + ch2*k2_eb + ch3*k3_eb + ch4*k4_eb + ch5*k5_eb + ch6*k6_eb;

    let te_lx = (ct1*k1_lx + ct2*k2_lx + ct3*k3_lx + ct4*k4_lx + ct5*k5_lx + ct6*k6_lx).abs();
    let te_ly = (ct1*k1_ly + ct2*k2_ly + ct3*k3_ly + ct4*k4_ly + ct5*k5_ly + ct6*k6_ly).abs();
    let te_lz = (ct1*k1_lz + ct2*k2_lz + ct3*k3_lz + ct4*k4_lz + ct5*k5_lz + ct6*k6_lz).abs();
    let te_eb = (ct1*k1_eb + ct2*k2_eb + ct3*k3_eb + ct4*k4_eb + ct5*k5_eb + ct6*k6_eb).abs();

    let te_max = te_lx.max(te_ly).max(te_lz).max(te_eb);

    let dtau_new: f64 = 0.9 * dtau * (epsilon/te_max).powf(1.0/5.0);

    if dlx5.is_nan() || dly5.is_nan() || dlz5.is_nan() || deb5.is_nan() {
        RKResult::Err("Found NaN")
    }
    else if te_eb > epsilon {
        RKResult::Repeat(dtau_new)
    }
    else {
        RKResult::Ok((lx + dlx5, ly + dly5, lz + dlz5, eb + deb5, dtau_new))
    }

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
        panic!("Invalid quadrant for ({},{})", x, y);
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
            return Result::Ok(State::Prograde);
        }
        else if crossed_x && has_positive {
            return Result::Ok(State::Retrograde);
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
    _dtau: f64,
    lx_init: f64,
    ly_init: f64,
    lz_init: f64,
    eb_init: f64,
    gamma: f64,
    walltime: f64,
    epsilon: f64
) -> SimResult {
    let start = SystemTime::now();
    let end = start + Duration::from_secs_f64(walltime);
    let mut tau = tau_init;
    let mut lx = lx_init;
    let mut ly = ly_init;
    let mut lz = lz_init;
    let mut eb = eb_init;
    let mut dtau = _dtau;

    let mut tau_arr = vec![tau_init];
    let mut lx_arr = vec![lx_init];
    let mut ly_arr = vec![ly_init];
    let mut lz_arr = vec![lz_init];
    let mut eb_arr = vec![eb_init];

    let mut i = get_i(lx, ly, lz);
    let mut omega = get_omega(lx, ly);
    assert!(!i.is_nan(),"Initial i is NaN");
    assert!(!omega.is_nan(),"Initial omega is NaN");
    
    let mut x = i * omega.cos();
    let mut y = i * omega.sin();
    let mut quad = get_quadrant(x, y);

    let mut hist = HashSet::<QuadrantTransition>::new();
    let mut current_state = State::Unknown;

    

    while SystemTime::now() < end {
        let res = rk4(tau, dtau, lx, ly, lz, eb, gamma, epsilon);
        match res {
            RKResult::Ok((lx_new,ly_new,lz_new,eb_new,dtau_new)) => {
                lx = lx_new;
                ly = ly_new;
                lz = lz_new;
                eb = eb_new;
                dtau = dtau_new;
                tau += dtau;
                tau_arr.push(tau);
                lx_arr.push(lx);
                ly_arr.push(ly);
                lz_arr.push(lz);
                eb_arr.push(eb);
                i = get_i(lx, ly, lz);
                omega = get_omega(lx, ly);
                assert!(!i.is_nan(),"i is NaN for ({}, {}, {}), eb = {}, gamma = {}, i_init = {}, omega_init = {}", lx, ly, lz, eb_init, gamma,get_i(lx_init, ly_init, lz_init),get_omega(lx_init, ly_init));
                assert!(!omega.is_nan(),"omega is NaN for ({}, {}, {}), eb = {}, gamma = {}", lx, ly, lz, eb_init, gamma);
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
            },
            RKResult::Repeat(dtau_new) => {
                dtau = dtau_new;
            },
            RKResult::Err(_) => {
                break
            }
        }


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
        let res = integrate(0.0, 0.1, lx, ly, lz, eb, gamma, 1.0,1e-10);
        let dtime = start_time.elapsed().unwrap().as_secs_f64();
        println!("{:?}", res);
        println!("Time integrating: {:?}", dtime);
        println!("Time for 1000: {:?}", dtime*1000.0);
        assert_eq!(res.state, State::Librating);
    }

}