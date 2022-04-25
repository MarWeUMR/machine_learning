use crate::ml::data_processing;
use crate::ml::qs::QuickArgSort;
use smartcore::dataset::breast_cancer;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::{mean_squared_error, roc_auc_score, accuracy};
use smartcore::model_selection::train_test_split;
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

    let y_hat = get_predictions(dataset, x_train, y_train, x_test, y_test);
    // println!("{:?}", y_hat);

    cancer();
}

fn cancer(){
    let cancer_data = breast_cancer::load_dataset();

    let x = DenseMatrix::from_array(
    cancer_data.num_samples,
    cancer_data.num_features,
    &cancer_data.data,
);
// These are our target class labels
let y = cancer_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Decision Tree
let y_hat_tree = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
    .and_then(|tree| tree.predict(&x_test)).unwrap();
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_tree));
}

fn get_target_column<'a>(set: Datasets) -> (&'a str, &'a str) {
    let result = match set {
        Datasets::Titanic => ("titanic", "Survived"),
        Datasets::Urban => ("urban", "class"),
        Datasets::Landcover => ("landcover", "Class_ID"),
    };
    result
}

fn get_predictions(
    dataset: &str,
    x_train: DenseMatrix<f32>,
    y_train: Vec<f32>,
    x_test: DenseMatrix<f32>,
    y_test: Vec<f32>,
) -> Vec<f32> {
    if dataset == "titanic" {
        println!("Selected model: Logistic Regression");
        let y_hat = LogisticRegression::fit(&x_train, &y_train, Default::default())
            .and_then(|lr| lr.predict(&x_test))
            .unwrap();
        println!("AUC: {}", roc_auc_score(&y_test, &y_hat));
        y_hat
    } else {
        println!("Selected model: Random forest classifier");
        println!("fitting...");

        let y_hat = RandomForestClassifier::fit(&x_train, &y_train, Default::default())
            .and_then(|rf| rf.predict(&x_test))
            .unwrap();

        println!("fitting done");

        println!("MSE: {}", accuracy(&y_test, &y_hat));
        y_hat
    }
}
