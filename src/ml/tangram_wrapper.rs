use crate::ml::data_processing::{get_data_matrix, get_tangram_matrix};

use super::data_processing;
use ndarray::{prelude::*, OwnedRepr};
use serde_json::json;
use std::{any::Any, path::Path};

use tangram_table::prelude::*;
use tangram_tree::{BinaryClassifierTrainOutput, Progress, RegressorTrainOutput, TrainOptions};
use tangram_zip::zip;

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

    // train the model
    // let train_output = TangramModel::train(
    //     x_train,
    //     y_train.clone(),
    //     &tangram_tree::TrainOptions {
    //         learning_rate: 0.1,
    //         max_leaf_nodes: 255,
    //         max_rounds: 100,
    //         ..Default::default()
    //     },
    //     Progress {
    //         kill_chip: &tangram_kill_chip::KillChip::default(),
    //         handle_progress_event: &mut |_| {},
    //     },
    // );

    let train_output = who_am_i(
        x_train,
        y_train.clone(),
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

    let printer: &BinaryClassifierTrainOutput =
        train_output.as_any()
         .downcast_ref::<BinaryClassifierTrainOutput>().unwrap();


    println!("yy {:?}", printer);

    // let yy: &BinaryClassifierTrainOutput = it.downcast_ref().unwrap();

    // Make predictions on the test data.
    let mut predictions = Array::zeros(y_test.len());
    // tangram_predict(x_test, o, &mut predictions);
    // tangram_predict(x_test, yy, &mut predictions);

    // ------------------------------------------------
    // REGRESSION METRICS
    // let mut metrics = tangram_metrics::RegressionMetrics::new();
    // metrics.update(tangram_metrics::RegressionMetricsInput {
    //     predictions: predictions.as_slice().unwrap(),
    //     labels: y_test_num.view().as_slice(),
    // });
    // let metrics = metrics.finalize();
    //
    // let output = json!({
    //     "mse": metrics.mse,
    // });
    // ------------------------------------------------

    // ------------------------------------------------
    // BINARY CLASSIFICATION METRICS
    tangram_evaluate(&mut predictions, y_test.clone());
}

fn get_dataset_info<'a>(set: Datasets) -> (&'a str, usize) {
    let result = match set {
        Datasets::Titanic => ("titanic", 13),
        Datasets::Urban => ("urban", 0),
        Datasets::Landcover => ("landcover", 12),
        Datasets::Boston => ("boston", 13),
        Datasets::Cancer => ("cancer", 29),
    };
    result
}

trait TangramModel: TMToAny {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> Self
    where
        Self: Sized;
}

impl TangramModel for RegressorTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> RegressorTrainOutput {
        let train_output = tangram_tree::Regressor::train(
            x_train.view(),
            y_train.as_number().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }
}

impl TangramModel for BinaryClassifierTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> BinaryClassifierTrainOutput {
        let train_output = tangram_tree::BinaryClassifier::train(
            x_train.view(),
            y_train.as_enum().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }
}

fn tangram_predict(
    x_test: Table,
    train_output: &BinaryClassifierTrainOutput,
    // train_output: Box<dyn TangramModel>,
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
) {
    // Make predictions on the test data.
    let x_test_array = x_test.to_rows(); // <- macht einen 2d array àla .to_ndarray()
                                         // let mut predictions = Array::zeros(y_test_typed.len());

    let t = train_output.as_any();

    train_output
        .model
        .predict(x_test_array.view(), predictions.view_mut());
}

fn tangram_evaluate(
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    y_test: TableColumn,
) {
    let input = zip!(predictions.iter(), y_test.as_enum().unwrap().iter())
        .map(|(probability, label)| (*probability, label.unwrap()))
        .collect();
    let auc_roc = tangram_metrics::AucRoc::compute(input);

    let output = json!({
        "auc_roc": auc_roc,
    });

    println!("{}", output);
}

fn who_am_i<'a>(
    x_train: Table,
    y_train: TableColumn,
    training_options: &TrainOptions,
    progress: Progress,
) -> Box<dyn Any> {
    let i = 2;

    let x = match i {
        2 => {
            let train_output = tangram_tree::BinaryClassifier::train(
                x_train.view(),
                y_train.as_enum().unwrap().view(),
                training_options,
                progress,
            );
            let e = E::A(train_output);

            e
        }
        _ => {
            let train_output = tangram_tree::Regressor::train(
                x_train.view(),
                y_train.as_number().unwrap().view(),
                training_options,
                progress,
            );
            let e = E::B(train_output);

            e
        }
    };

    // let train_output = tangram_tree::BinaryClassifier::train(
    //     x_train.view(),
    //     y_train.as_enum().unwrap().view(),
    //     training_options,
    //     progress,
    // );
    // let e = E::A(train_output);

    Box::new(x) as Box<dyn Any>
}

enum E {
    A(BinaryClassifierTrainOutput),
    B(RegressorTrainOutput),
}

pub trait TMToAny: 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> TMToAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
