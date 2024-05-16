pub mod solve;

use pyo3::prelude::*;
use numpy::PyArray1;


/// A Python module implemented in Rust.
#[pymodule]
fn _polar_disk_freq(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve::get_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(solve::init_xyz, m)?)?;

    #[pyfunction]
    fn integrate_py(
        py: Python,
        tau_init: f64,
        dtau: f64,
        lx_init: f64,
        ly_init: f64,
        lz_init: f64,
        eb_init: f64,
        gamma: f64,
        walltime: f64,
        epsilon: f64
    ) -> PyResult<(Py<PyArray1<f64>>,Py<PyArray1<f64>>,Py<PyArray1<f64>>,Py<PyArray1<f64>>,Py<PyArray1<f64>>,String)> {
        let result: solve::SimResult = solve::integrate(tau_init, dtau, lx_init, ly_init, lz_init, eb_init, gamma, walltime, epsilon);

        let state_str = match result.state {
            solve::State::Prograde => "p",
            solve::State::Retrograde => "r",
            solve::State::Librating => "l",
            solve::State::Unknown => "u",
        };
        PyResult::Ok((
            PyArray1::from_vec_bound(py,result.tau.to_vec()).into(),
            PyArray1::from_vec_bound(py,result.lx.to_vec()).into(),
            PyArray1::from_vec_bound(py,result.ly.to_vec()).into(),
            PyArray1::from_vec_bound(py,result.lz.to_vec()).into(),
            PyArray1::from_vec_bound(py,result.eb.to_vec()).into(),
            state_str.to_string()
        ))
    }

    m.add_function(wrap_pyfunction!(integrate_py, m)?)?;

    Ok(())
}
