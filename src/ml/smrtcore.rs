use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;

use crate::ml::data_processing;

pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
}

pub fn run(set: Datasets) {
    let (dataset, target_column) = get_target_column(set);

    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_data_matrix(dataset, target_column);

    let x = DenseMatrix::from_vec(
        x_train_array.nrows(),
        x_train_array.ncols(),
        x_train_array.into_raw_vec().as_slice(),
    );
    let y = y_train_array.into_raw_vec();
    let yy = get_predictions(dataset, x, y);

    // let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
    // let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    // let y_hat = lr.predict(&x).unwrap();

    println!("x_train_array: {:?}", yy);
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
    let mut yy = Vec::new();

    if dataset == "titanic" {
        println!("Selected model: Logistic Regression");
        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
        let y_hat = lr.predict(&x).unwrap();
        yy = y_hat.clone();
    } else {
        println!("Selected model: Random forest classifier");
        let rf = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();
        let y_hat = rf.predict(&x).unwrap();
        yy = y_hat.clone();
    }

    yy
}
