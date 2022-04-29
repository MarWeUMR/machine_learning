use std::path::Path;

use ndarray::Array;
use polars::{
    datatypes::Float32Type,
    io::SerReader,
    prelude::{CsvReader, DataFrame, NamedFrom, Series},
};
use pyo3::types::PyModule;
use tangram_table::Table;

pub fn run_through_python(dataset: &str) {
    // use python to preprocess data
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let py_mod =
        PyModule::from_code(py, include_str!("../test.py"), "filename.py", "modulename").unwrap();

    let py_load_data = py_mod.getattr("load_data").unwrap();
    py_load_data.call1((dataset,)).unwrap();
}

pub fn get_multiclass_label_count(
    dataset: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
) -> u32 {
    let target = polars::series::Series::new("target", dataset.as_slice().unwrap());

    let count = target.n_unique().unwrap();
    count as u32
}

pub fn get_tangram_matrix(dataset: &str, target_column_idx: usize) -> (Table, Table, tangram_table::TableColumn, tangram_table::TableColumn) {
    let train_path = &format!("datasets/{dataset}/train_data_enc.csv");
    let test_path = &format!("datasets/{dataset}/test_data_enc.csv");
    let csv_file_path_train = Path::new(train_path);
    let csv_file_path_test = Path::new(test_path);

    let mut x_train =
        Table::from_path(csv_file_path_train, Default::default(), &mut |_| {}).unwrap();
    let mut x_test = Table::from_path(csv_file_path_test, Default::default(), &mut |_| {}).unwrap();

    let y_train = x_train.columns_mut().remove(target_column_idx);
    let y_test = x_test.columns_mut().remove(target_column_idx);
    let y_train_num = y_train.as_number().unwrap();
    let y_test_num = y_test.as_number().unwrap();

    (x_train, x_test, y_train, y_test)
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
