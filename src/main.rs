extern crate xgboost;
use ndarray::prelude::*;

use xgboost::{parameters, Booster, DMatrix};

use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};

use polars::prelude::*;

mod ml;



fn main() {
    // ml::xgbindings::run(ml::xgbindings::Datasts::Landcover);
    // ml::xgbindings::run(ml::xgbindings::Datasts::Titanic);
    // ml::xgbindings::run(ml::xgbindings::Datasts::Urban);

    ml::smrtcore::run();
}
