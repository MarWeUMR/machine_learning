extern crate xgboost;
use ndarray::prelude::*;

use xgboost::{parameters, Booster, DMatrix};

use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};

use polars::prelude::*;

fn main() {
    // use python to preprocess data
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let _py_mod =
        PyModule::from_code(py, include_str!("test.py"), "filename.py", "modulename").unwrap();

    // read preprocessed data to rust
    let x_train_frame: DataFrame = CsvReader::from_path("datasets/urban/train_data_enc.csv")
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();

    let x_test_frame: DataFrame = CsvReader::from_path("datasets/urban/test_data_enc.csv")
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();

    let y_train_frame =
        DataFrame::new(vec![x_train_frame.column("class_enc").unwrap().clone()]).unwrap();
    let y_train_array = y_train_frame.to_ndarray::<Float32Type>().unwrap();
    x_train_frame.drop("class_enc").unwrap();

    let y_test_frame =
        DataFrame::new(vec![x_test_frame.column("class_enc").unwrap().clone()]).unwrap();
    let y_test_array = y_test_frame.to_ndarray::<Float32Type>().unwrap();
    x_test_frame.drop("class_enc").unwrap();

    let x_train_array: Array<f32, _> = x_train_frame.to_ndarray::<Float32Type>().unwrap();
    let x_test_array: Array<f32, _> = x_test_frame.to_ndarray::<Float32Type>().unwrap();

    let num_rows_train = &x_train_array.len();
    let num_rows_test = &x_test_array.len();

    // println!("{:?}", x_train_array.shape()[0]);

    let train_shape = x_train_array.raw_dim();
    let test_shape = x_test_array.raw_dim();

    println!("{:?}", train_shape);

    let mut dtrain = DMatrix::from_dense(
        x_train_array
            .into_shape(train_shape[0] * train_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        *num_rows_train,
    )
    .unwrap();

    let mut dtest = DMatrix::from_dense(
        x_test_array
            .into_shape(test_shape[0] * test_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        *num_rows_test,
    )
    .unwrap();

    dtrain
        .set_labels(y_train_array.as_slice().unwrap())
        .unwrap();

    dtest.set_labels(y_test_array.as_slice().unwrap()).unwrap();

    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build()
        .unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(2)
        .eta(1.0)
        .build()
        .unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build()
        .unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // overall configuration for training/evaluation
    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain) // dataset to train with
        .boost_rounds(2) // number of training iterations
        .booster_params(booster_params) // model parameters
        .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
