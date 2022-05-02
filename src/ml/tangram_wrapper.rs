use crate::ml::data_processing::{get_data_matrix, get_tangram_matrix};

use super::data_processing;
use ndarray::{prelude::*, OwnedRepr};
use serde_json::json;
use std::{any::Any, path::Path};

use tangram_table::prelude::*;
use tangram_tree::{
    BinaryClassifierTrainOutput, MulticlassClassifierTrainOutput, Progress, RegressorTrainOutput,
    TrainOptions,
};
use tangram_zip::zip;

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Numeric,
    Binary,
    MultiClass,
}

#[derive(Debug, Clone, Copy)]
pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
    Boston,
    Cancer,
}

pub fn run(set: Datasets) {
    let (dataset, target_column_idx, model_type) = get_dataset_info(set);

    // use python to preprocess data
    data_processing::run_through_python(dataset);

    let (x_train, x_test, y_train, y_test) = get_tangram_matrix(dataset, target_column_idx);

    // -------------------------------------------------------------------
    // Train the model using the correct algorithm for the given dataset
    // ---------------------------------------------------------
    // ....
    let train_output = tangram_train_model(
        model_type,
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

    // -------------------------------------
    // Make predictions on the test data.
    // --------------------------
    // ....
    let mut predictions = Array::zeros(y_test.len());
    tangram_predict(model_type, x_test, train_output, &mut predictions);

    // -------------------------------------
    // Evaluate the model with the appropriate metrics
    // --------------------------
    // ....

    tangram_evaluate(model_type, &mut predictions, y_test.clone());
}

fn get_dataset_info<'a>(set: Datasets) -> (&'a str, usize, ModelType) {
    let result = match set {
        Datasets::Titanic => ("titanic", 13, ModelType::Binary),
        Datasets::Urban => ("urban", 0, ModelType::MultiClass),
        Datasets::Landcover => ("landcover", 12, ModelType::MultiClass),
        Datasets::Boston => ("boston", 13, ModelType::Numeric),
        Datasets::Cancer => ("cancer", 30, ModelType::Binary),
    };
    result
}

trait TangramModel {
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TangramModel for MulticlassClassifierTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> MulticlassClassifierTrainOutput {
        let train_output = tangram_tree::MulticlassClassifier::train(
            x_train.view(),
            y_train.as_enum().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn tangram_predict(
    model_type: ModelType,
    x_test: Table,
    train_output: Box<dyn TangramModel>,
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
) {
    println!("Predicting test data...");

    match model_type {
        ModelType::Numeric => {
            let x = train_output
                .as_any()
                .downcast_ref::<RegressorTrainOutput>()
                .expect("Couldn't downcast to RegressorTrainOutput");

            x.model
                .predict(x_test.to_rows().view(), predictions.view_mut());
        }
        ModelType::Binary => {
            let x = train_output
                .as_any()
                .downcast_ref::<BinaryClassifierTrainOutput>()
                .expect("Couldn't downcast to BinaryClassifierTrainOutput");

            x.model
                .predict(x_test.to_rows().view(), predictions.view_mut());
        }
        ModelType::MultiClass => todo!(),
    };
}

fn tangram_evaluate(
    model_type: ModelType,
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    y_test: TableColumn,
) {
    println!("Evaluating model...");
    match model_type {
        ModelType::Numeric => {
            let mut metrics = tangram_metrics::RegressionMetrics::new();
            metrics.update(tangram_metrics::RegressionMetricsInput {
                predictions: predictions.as_slice().unwrap(),
                labels: y_test.as_number().unwrap().view().as_slice(),
            });
            let metrics = metrics.finalize();

            let output = json!({
                "mse": metrics.mse,
            });
            println!("{}", output);
        }
        ModelType::Binary => {
            let input = zip!(predictions.iter(), y_test.as_enum().unwrap().iter())
                .map(|(probability, label)| (*probability, label.unwrap()))
                .collect();
            let auc_roc = tangram_metrics::AucRoc::compute(input);

            let output = json!({
                "auc_roc": auc_roc,
            });

            println!("{}", output);
        }
        ModelType::MultiClass => todo!(),
    };
}

fn tangram_train_model(
    model_type: ModelType,
    x_train: Table,
    y_train: TableColumn,
    training_options: &TrainOptions,
    progress: Progress,
) -> Box<dyn TangramModel> {
    println!("Training model...");
    println!("{:?}", y_train);
    match model_type {
        ModelType::Binary => {
            println!("returning BinaryClassifierTrainOutput");
            let train_output = tangram_tree::BinaryClassifier::train(
                x_train.view(),
                y_train.as_enum().unwrap().view(),
                training_options,
                progress,
            );

            Box::new(train_output)
        }
        ModelType::Numeric => {
            println!("returning RegressorTrainOutput");
            let train_output = tangram_tree::Regressor::train(
                x_train.view(),
                y_train.as_number().unwrap().view(),
                training_options,
                progress,
            );
            Box::new(train_output)
        }
        ModelType::MultiClass => {
            let train_output = tangram_tree::MulticlassClassifier::train(
                x_train.view(),
                y_train.as_enum().unwrap().view(),
                training_options,
                progress,
            );
            Box::new(train_output)
        }
    }
}
