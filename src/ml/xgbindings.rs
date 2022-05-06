use polars::{io::SerReader, prelude::*};
use xgboost_bindings::{
    parameters::{self, learning::LearningTaskParametersBuilder, BoosterParametersBuilder},
    Booster, DMatrix,
};

use crate::ml::data_processing;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
    Boston,
    Cancer,
    Iris,
}

pub fn run(set: Datasets) {
    // one hot columns
    let (dataset, path, target_column, ohe_cols, label_encode_cols) = get_dataset_metadata(set);

    println!("Working on dataset: {}", dataset);

    // use python to preprocess data
    // data_processing::run_through_python(dataset);

    // read preprocessed data to rust
    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_xg_matrix(path, target_column, ohe_cols, label_encode_cols);

    let train_shape = x_train_array.raw_dim();
    let test_shape = x_test_array.raw_dim();

    let num_rows_train = train_shape[0];
    let num_rows_test = test_shape[0];

    // let mut x_train = DMatrix::load("datasets/urban/train_data_xg.csv").unwrap();
    // let mut x_test = DMatrix::load("datasets/urban/test_data_xg.csv").unwrap();

    dbg!(&x_train_array);

    let mut x_train = DMatrix::from_dense(
        x_train_array
            .into_shape(train_shape[0] * train_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_train,
    )
    .unwrap();

    let mut x_test = DMatrix::from_dense(
        x_test_array
            .into_shape(test_shape[0] * test_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_test,
    )
    .unwrap();

    x_train
        .set_labels(y_train_array.as_slice().unwrap())
        .unwrap();

    x_test.set_labels(y_test_array.as_slice().unwrap()).unwrap();

    let _xg_classifier = get_objective(set, y_train_array.clone());

    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&x_train)
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    let preds = bst.predict(&x_test).unwrap();

    let labels = x_test.get_labels().unwrap();
    println!(
        "First 3 predicted labels: {} {} {}",
        labels[0], labels[1], labels[2]
    );

    // TODO: add meaningful evaluation metrics
    // print error rate
    let num_correct: usize = preds.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).sum();
    println!(
        "error={} ({}/{} correct)",
        num_correct as f32 / preds.len() as f32,
        num_correct,
        preds.len()
    );
}

fn get_dataset_metadata<'a>(
    set: Datasets,
) -> (&'a str, &'a str, &'a str, Vec<&'a str>, Vec<&'a str>) {
    let result = match set {
        Datasets::Titanic => (
            "titanic",
            "datasets/titanic/train_data.csv",
            "target",
            vec!["sex", "cabin", "embarked", "home.dest"],
            vec![],
        ),
        Datasets::Landcover => (
            "landcover",
            "datasets/landcover/train_data.csv",
            "Class_ID",
            vec![],
            vec![],
        ),
        Datasets::Urban => (
            "urban",
            "datasets/urban/data.csv",
            "class",
            vec![],
            vec!["class"],
        ),
        Datasets::Boston => (
            "boston",
            "datasets/boston/train_data.csv",
            "MEDV",
            vec![],
            vec![],
        ),
        Datasets::Cancer => (
            "cancer",
            "datasets/cancer/train_data.csv",
            "target",
            vec![],
            vec![],
        ),
        Datasets::Iris => ("iris", "datasets/iris/data.csv", "target", vec![], vec![]),
    };
    result
}

fn get_objective<'a>(
    set: Datasets,
    y_train: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
) -> parameters::learning::Objective {
    let n_unique = data_processing::get_multiclass_label_count(y_train.clone());

    let result = match set {
        Datasets::Titanic => {
            println!("BinaryLogistic chosen");
            parameters::learning::Objective::BinaryLogistic
        }
        Datasets::Urban => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftmax(n_unique)
        }
        Datasets::Landcover => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftmax(n_unique)
        }
        Datasets::Boston => {
            println!("LinReg chosen");
            parameters::learning::Objective::RegLinear
        }
        Datasets::Cancer => {
            println!("BinaryLogistic chosen");
            parameters::learning::Objective::BinaryLogistic
        }
        Datasets::Iris => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftmax(n_unique)
        }
    };
    result
}
