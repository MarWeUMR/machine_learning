// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::math::distance::*;
use smartcore::neighbors::knn_classifier::KNNClassifier;

use crate::ml::data_processing;

pub fn run() {
    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_data_matrix("titanic", "Survived");
    
    let x = DenseMatrix::from_vec(x_train_array.nrows(), x_train_array.ncols(), x_train_array.into_raw_vec().as_slice());
    let y = y_train_array.into_raw_vec();

    // let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
    let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    let y_hat = lr.predict(&x).unwrap();

    println!("x_train_array: {:?}", y);

}
