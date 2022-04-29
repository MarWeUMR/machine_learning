use crate::ml::data_processing::{get_data_matrix, get_tangram_matrix};

use super::data_processing;
use ndarray::prelude::*;
use serde_json::json;
use std::path::Path;

use tangram_table::prelude::*;
use tangram_tree::Progress;

#[derive(Debug, Clone, Copy)]
pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
    Boston,
    Cancer,
}

pub fn run(set: Datasets) {
    let (dataset, target_column_idx) = get_dataset_info(set);

    // use python to preprocess data
    data_processing::run_through_python(dataset);

    let (x_train, x_test, y_train, y_test) = get_tangram_matrix(dataset, target_column_idx);

    let y_train_num = y_train.as_number().unwrap();
    let y_test_num = y_test.as_number().unwrap();

    // Train the model.
    let train_output = tangram_tree::Regressor::train(
        x_train.view(),
        y_train_num.view(),
        &tangram_tree::TrainOptions {
            learning_rate: 0.1,
            max_leaf_nodes: 255,
            max_rounds: 100,
            ..Default::default()
        },
        Progress {
            kill_chip: &tangram_kill_chip::KillChip::default(),
            handle_progress_event: &mut |_| {},
        },
    );

    // Make predictions on the test data.
    let x_test_array = x_test.to_rows(); // <- macht einen 2d array àla .to_ndarray()
    let mut predictions = Array::zeros(y_test_num.len());
    train_output
        .model
        .predict(x_test_array.view(), predictions.view_mut());

    // Compute metrics.
    let mut metrics = tangram_metrics::RegressionMetrics::new();
    metrics.update(tangram_metrics::RegressionMetricsInput {
        predictions: predictions.as_slice().unwrap(),
        labels: y_test_num.view().as_slice(),
    });
    let metrics = metrics.finalize();

    let output = json!({
        "mse": metrics.mse,
    });
    println!("{}", output);
}

fn get_dataset_info<'a>(set: Datasets) -> (&'a str, usize) {
    let result = match set {
        Datasets::Titanic => ("titanic", 10),
        Datasets::Urban => ("urban", 0),
        Datasets::Landcover => ("landcover", 12),
        Datasets::Boston => ("boston", 13),
        Datasets::Cancer => ("cancer", 29),
    };
    result
}
