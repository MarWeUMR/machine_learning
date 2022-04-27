use crate::ml::data_processing;
use smartcore::dataset::breast_cancer;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;

pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
    Boston,
    Cancer,
}

pub fn run(set: Datasets) {
    // ------------------

    let (dataset, target_column) = get_target_column(set);

    // use python to preprocess data
    data_processing::run_through_python(dataset);

    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_data_matrix(dataset, target_column);

    let x_train = DenseMatrix::from_vec(
        x_train_array.nrows(),
        x_train_array.ncols(),
        &x_train_array.into_raw_vec(),
    );

    let x_test = DenseMatrix::from_vec(
        x_test_array.nrows(),
        x_test_array.ncols(),
        &x_test_array.into_raw_vec(),
    );

    let y_train = y_train_array.into_raw_vec();
    let y_test = y_test_array.into_raw_vec();

    println!(
        "type of target data: {}, {}, {}...",
        &y_train.get(0).unwrap(),
        &y_train.get(1).unwrap(),
        &y_train.get(2).unwrap()
    );

    let y_hat = get_predictions(dataset, x_train, y_train, x_test, y_test.clone());
    println!(
        "First 3 predictions:\n \t{} vs. {}\n \t{} vs. {}\n \t{} vs {}",
        y_hat.get(0).unwrap(),
        y_test.get(0).unwrap(),
        y_hat.get(1).unwrap(),
        y_test.get(1).unwrap(),
        y_hat.get(2).unwrap(),
        y_test.get(2).unwrap(),
    );
}

fn get_target_column<'a>(set: Datasets) -> (&'a str, &'a str) {
    let target_col_name = match set {
        Datasets::Titanic => ("titanic", "Survived"),
        Datasets::Urban => ("urban", "class"),
        Datasets::Landcover => ("landcover", "Class_ID"),
        Datasets::Boston => ("boston", "MEDV"),
        Datasets::Cancer => ("cancer", "target"),
    };
    target_col_name
}

fn get_predictions(
    dataset: &str,
    x_train: DenseMatrix<f32>,
    y_train: Vec<f32>,
    x_test: DenseMatrix<f32>,
    y_test: Vec<f32>,
) -> Vec<f32> {
    // define dataset types
    let binary_data = vec!["titanic", "cancer"];
    let multiclass_data = vec!["landcover", "urban"];
    let numeric_data = vec!["boston"];

    if binary_data.contains(&dataset) {
        println!("Using DecisionTreeClassifier");
        // vvv ---- funktioniert nicht...
        let y_hat = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
            // let y_hat = LogisticRegression::fit(&x_train, &y_train, Default::default())
            .and_then(|model| model.predict(&x_test))
            .unwrap();

        println!("AUC: {}", roc_auc_score(&y_test, &y_hat));
        y_hat
    } else if numeric_data.contains(&dataset) {
        println!("Using RandomForestRegressor");
        let y_hat = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
            .and_then(|model| model.predict(&x_test))
            .unwrap();

        println!("MSE: {}", mean_squared_error(&y_test, &y_hat));
        y_hat
    } else if multiclass_data.contains(&dataset) {
        println!("Using RandomForestClassifier");
        let y_hat = RandomForestClassifier::fit(&x_train, &y_train, Default::default())
            .and_then(|model| model.predict(&x_test))
            .unwrap();

        println!("Accuracy: {}", accuracy(&y_test, &y_hat));
        y_hat
    } else {
        println!("dataset not defined yet");
        vec![0.0]
    }
}
