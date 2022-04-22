extern crate xgboost;
use ndarray::prelude::*;

use xgboost::{parameters, Booster, DMatrix};

use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyModule},
};

use polars::prelude::*;

fn main() {
    // specify main ingredients
    let dataset = "landcover";
    let target_column = "Class_ID";

    // use python to preprocess data
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let py_mod =
        PyModule::from_code(py, include_str!("test.py"), "filename.py", "modulename").unwrap();

    let py_load_data = py_mod.getattr("load_data").unwrap();
    py_load_data.call1((dataset,)).unwrap();

    // read preprocessed data to rust

    let x_train_frame: DataFrame =
        CsvReader::from_path(format!("datasets/{dataset}/test_data_enc.csv"))
            .unwrap()
            .infer_schema(None)
            .has_header(true)
            .finish()
            .unwrap();

    let x_test_frame: DataFrame =
        CsvReader::from_path(format!("datasets/{dataset}/test_data_enc.csv"))
            .unwrap()
            .infer_schema(None)
            .has_header(true)
            .finish()
            .unwrap();

    let y_train_frame =
        DataFrame::new(vec![x_train_frame.column(target_column).unwrap().clone()]).unwrap();
    let y_train_array = y_train_frame.to_ndarray::<Float32Type>().unwrap();
    x_train_frame.drop(target_column).unwrap();

    let y_test_frame =
        DataFrame::new(vec![x_test_frame.column(target_column).unwrap().clone()]).unwrap();
    let y_test_array = y_test_frame.to_ndarray::<Float32Type>().unwrap();
    x_test_frame.drop(target_column).unwrap();

    let x_train_array: Array<f32, _> = x_train_frame.to_ndarray::<Float32Type>().unwrap();
    let x_test_array: Array<f32, _> = x_test_frame.to_ndarray::<Float32Type>().unwrap();

    let train_shape = x_train_array.raw_dim();
    let test_shape = x_test_array.raw_dim();

    let num_rows_train = train_shape[0];
    let num_rows_test = test_shape[0];

    println!("{:?}", train_shape);

    let mut dtrain = DMatrix::from_dense(
        x_train_array
            .into_shape(train_shape[0] * train_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_train,
    )
    .unwrap();

    let mut dtest = DMatrix::from_dense(
        x_test_array
            .into_shape(test_shape[0] * test_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_test,
    )
    .unwrap();

    dtrain
        .set_labels(y_train_array.as_slice().unwrap())
        .unwrap();

    dtest.set_labels(y_test_array.as_slice().unwrap()).unwrap();

    /*
    ---------------------------------------------------------------------------------
     *  SPECIFY XGBOOST PARAMETERS
     *
     *  ---------------------------------------------------------------------------------
    */

    // let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
    //     .objective(parameters::learning::Objective::BinaryLogistic)
    //     .build()
    //     .unwrap();

    // configure the tree-based learning model's parameters
    // let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
    //     .max_depth(5)
    //     .eta(1.0)
    //     .num_parallel_tree(100)
    //     .colsample_bytree(0.8)
    //     .build()
    //     .unwrap();

    // overall configuration for Booster
    // let booster_params = parameters::BoosterParametersBuilder::default()
    //     .booster_type(parameters::BoosterType::Tree(tree_params))
    //     .learning_params(learning_params)
    //     .verbose(true)
    //     .build()
    //     .unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // overall configuration for training/evaluation
    // let params = parameters::TrainingParametersBuilder::default()
    //     .dtrain(&dtrain) // dataset to train with
    //     .boost_rounds(2) // number of training iterations
    //     .booster_params(booster_params) // model parameters
    //     .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
    //     .build()
    //     .unwrap();

    // let prms = parameters::TrainingParametersBuilder::default()
    //     .dtrain(&dtrain)
    //     .build()
    //     .unwrap();

    // RANDOM FOREST SETUP -----------------------------------

    let rf_tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .subsample(0.8)
        .max_depth(5)
        .eta(1.0) // aka learning_rate
        .colsample_bytree(0.8)
        .num_parallel_tree(100).tree_method(parameters::tree::TreeMethod::Hist)
        .build()
        .unwrap();

    let rf_learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build()
        .unwrap();

    let rf_booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(rf_tree_params))
        .learning_params(rf_learning_params)
        .build()
        .unwrap();

    let rf_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .boost_rounds(1)
        .booster_params(rf_booster_params)
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&rf_params).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
    let preds = bst.predict(&dtest).unwrap();

    let labels = dtest.get_labels().unwrap();
    println!(
        "First 3 predicted labels: {} {} {}",
        labels[0], labels[1], labels[2]
    );

    // print error rate
    let num_correct: usize = preds.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).sum();
    println!(
        "error={} ({}/{} correct)",
        num_correct as f32 / preds.len() as f32,
        num_correct,
        preds.len()
    );
}
