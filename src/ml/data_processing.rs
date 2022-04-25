use ndarray::Array;
use polars::{
    datatypes::Float32Type,
    io::SerReader,
    prelude::{CsvReader, DataFrame},
};
use pyo3::types::PyModule;

pub fn run_through_python(dataset: &str) {
    // use python to preprocess data
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let py_mod =
        PyModule::from_code(py, include_str!("../test.py"), "filename.py", "modulename").unwrap();

    let py_load_data = py_mod.getattr("load_data").unwrap();
    py_load_data.call1((dataset,)).unwrap();
}

pub fn get_data_matrix(
    dataset: &str,
    target_column: &str,
) -> (
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
) {
    // read preprocessed data to rust
    let mut x_train_frame: DataFrame =
        CsvReader::from_path(format!("datasets/{dataset}/train_data_enc.csv"))
            .unwrap()
            .infer_schema(None)
            .has_header(true)
            .finish()
            .unwrap();

    let mut x_test_frame: DataFrame =
        CsvReader::from_path(format!("datasets/{dataset}/test_data_enc.csv"))
            .unwrap()
            .infer_schema(None)
            .has_header(true)
            .finish()
            .unwrap();

    let y_train_frame =
        DataFrame::new(vec![x_train_frame.column(target_column).unwrap().clone()]).unwrap();
    let y_train_array = y_train_frame.to_ndarray::<Float32Type>().unwrap();
    x_train_frame.drop_in_place(target_column).unwrap();

    let y_test_frame =
        DataFrame::new(vec![x_test_frame.column(target_column).unwrap().clone()]).unwrap();
    let y_test_array = y_test_frame.to_ndarray::<Float32Type>().unwrap();
    x_test_frame.drop_in_place(target_column).unwrap();

    let x_train_array: Array<f32, _> = x_train_frame.to_ndarray::<Float32Type>().unwrap();
    let x_test_array: Array<f32, _> = x_test_frame.to_ndarray::<Float32Type>().unwrap();

    // println!("{:?}", x_train_frame);
    // println!("{:?}", x_train_frame.transpose());

    (x_train_array, x_test_array, y_train_array, y_test_array)
}
