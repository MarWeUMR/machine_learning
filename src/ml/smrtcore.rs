use crate::ml::data_processing;
use crate::ml::qs::QuickArgSort;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;

pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
}

pub fn run(set: Datasets) {
    // ------------------

    let (dataset, target_column) = get_target_column(set);

    // use python to preprocess data
    data_processing::run_through_python(dataset);

    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_data_matrix(dataset, target_column);

    let z = x_train_array.clone().t().to_owned();
    let x = DenseMatrix::from_vec(
        z.ncols(),
        z.nrows(),
        &z.t().to_owned().into_raw_vec(),
    );

    println!("{:?}", &x);
    println!("{:?}", &z);

    let y = y_train_array.into_raw_vec();
    let y_hat = get_predictions(dataset, x, y);
}

fn get_target_column<'a>(set: Datasets) -> (&'a str, &'a str) {
    let result = match set {
        Datasets::Titanic => ("titanic", "Survived"),
        Datasets::Urban => ("urban", "class"),
        Datasets::Landcover => ("landcover", "Class_ID"),
    };
    result
}

fn get_predictions(dataset: &str, x: DenseMatrix<f32>, y: Vec<f32>) -> Vec<f32> {
    if dataset == "titanic" {
        println!("Selected model: Logistic Regression");
        let lr = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();
        let y_hat = lr.predict(&x).unwrap();
        y_hat
    } else {
        println!("Selected model: Random forest classifier");
        println!("fitting...");
        let rf = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();
        println!("fitting done");
        let y_hat = rf.predict(&x).unwrap();
        y_hat
    }
}
