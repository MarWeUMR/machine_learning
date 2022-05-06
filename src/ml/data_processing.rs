use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use polars::{
    datatypes::{Float32Type, IdxCa},
    io::SerReader,
    prelude::*,
};
use pyo3::types::PyModule;
use tangram_table::{Table, TableColumnType};

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
    dataset: ndarray::ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> u32 {
    let target = polars::series::Series::new("target", dataset.as_slice().unwrap());

    let count = target.n_unique().unwrap();
    count as u32
}

pub fn get_xg_matrix(
    path: &str,
    target_column: &str,
    one_hot_encode_columns: Vec<&str>,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    let mut df = load_dataframe_from_file(path);

    let (x_train, x_test, y_train, y_test) =
        split_data(df.clone(), target_column, &one_hot_encode_columns);

    (x_train, x_test, y_train, y_test)
}

pub fn label_encode(col: Series) -> Vec<usize> {
    let uniques = col.unique().unwrap();
    let unique_count = uniques.len();

    let mut hm: HashMap<String, usize> = HashMap::new();

    for (i, elem) in uniques.iter().enumerate() {
        println!("{}, {}", elem, i);
        hm.insert(String::from(elem.to_string()), i);
    }

    let v: Vec<usize> = col
        .iter()
        .map(|elem| *hm.get(&elem.to_string()).unwrap())
        .collect();
    dbg!(&v);
    v
}

fn one_hot_encode_dataframe(df: &mut DataFrame, categorical_columns: &[&str]) {
    for col in categorical_columns.iter() {
        println!("encoding: {}", col);

        let col_pre_encoding = df.drop_in_place(col).unwrap();
        let col_ohe = col_pre_encoding.to_dummies().unwrap();

        df.hstack_mut(col_ohe.get_columns()).unwrap();
    }
}

pub fn one_hot_encode_column(path: &str, target_column: &str) {
    let mut df = load_dataframe_from_file(path);

    one_hot_encode_dataframe(&mut df, &["target"]);

    println!("{}", df);

    // let target: Series = df.drop_in_place("target").unwrap();
    // let ohe = target.to_dummies().unwrap();
}

pub fn get_tangram_matrix(
    dataset: &str,
    target_column_idx: usize,
) -> (
    Table,
    Table,
    tangram_table::TableColumn,
    tangram_table::TableColumn,
) {
    let train_path = &format!("datasets/{dataset}/train_data.csv");
    let test_path = &format!("datasets/{dataset}/test_data.csv");
    let csv_file_path_train = Path::new(train_path);
    let csv_file_path_test = Path::new(test_path);

    // ------------------------------------
    let target_variants = ["1", "2", "3", "4", "5"]
        .iter()
        .map(ToString::to_string)
        .collect();
    let options = tangram_table::FromCsvOptions {
        column_types: Some(BTreeMap::from([(
            "Class_ID".to_owned(),
            TableColumnType::Enum {
                variants: target_variants,
            },
        )])),
        ..Default::default()
    };

    // ------------------------------------

    let mut x_train = Table::from_path(csv_file_path_train, options.clone(), &mut |_| {}).unwrap();
    let mut x_test = Table::from_path(csv_file_path_test, options.clone(), &mut |_| {}).unwrap();

    let y_train = x_train.columns_mut().remove(target_column_idx);
    let y_test = x_test.columns_mut().remove(target_column_idx);

    (x_train, x_test, y_train, y_test)
}

fn split_data(
    df: DataFrame,
    target_column: &str,
    one_hot_encode_columns: &Vec<&str>,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    let n_rows = df.shape().0 as u32;
    let split_position = f32::floor(0.8 * n_rows as f32) as u32;

    let first_part: Vec<_> = (0..split_position).collect();
    let first_part_slice = first_part.as_slice();
    let second_part: Vec<_> = (split_position..n_rows - 1).collect();
    let second_part_slice = second_part.as_slice();

    let idx_train = IdxCa::new("idx", first_part_slice);
    let idx_test = IdxCa::new("idx", second_part_slice);

    let mut x_train: DataFrame = df.take(&idx_train).unwrap();
    let mut x_test: DataFrame = df.take(&idx_test).unwrap();

    // one_hot_encode_columns
    one_hot_encode_dataframe(&mut x_train, &one_hot_encode_columns);
    one_hot_encode_dataframe(&mut x_test, &one_hot_encode_columns);

    let y_test: DataFrame =
        DataFrame::new(vec![x_test.drop_in_place(target_column).unwrap()]).unwrap();
    let y_train: DataFrame =
        DataFrame::new(vec![x_train.drop_in_place(target_column).unwrap()]).unwrap();

    println!("{}", x_train);
    (
        x_train.to_ndarray::<Float32Type>().unwrap(),
        x_test.to_ndarray::<Float32Type>().unwrap(),
        y_train.to_ndarray::<Float32Type>().unwrap(),
        y_test.to_ndarray::<Float32Type>().unwrap(),
    )
}

fn load_dataframe_from_file(path: &str) -> DataFrame {
    let df: DataFrame = CsvReader::from_path(path)
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();

    df
}

pub fn get_data_matrix(
    dataset: &str,
    target_column: &str,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    // read preprocessed data to rust
    let mut x_train_frame =
        load_dataframe_from_file(format!("datasets/{dataset}/train_data.csv").as_str());
    let mut x_test_frame =
        load_dataframe_from_file(format!("datasets/{dataset}/test_data.csv").as_str());

    let y_train_frame = DataFrame::new(vec![x_train_frame
        .drop_in_place(target_column)
        .unwrap()
        .clone()])
    .unwrap();
    let y_train_array = y_train_frame.to_ndarray::<Float32Type>().unwrap();
    // x_train_frame.drop_in_place(target_column).unwrap();

    let y_test_frame = DataFrame::new(vec![x_test_frame
        .drop_in_place(target_column)
        .unwrap()
        .clone()])
    .unwrap();
    let y_test_array = y_test_frame.to_ndarray::<Float32Type>().unwrap();
    // x_test_frame.drop_in_place(target_column).unwrap();

    let x_train_array: Array<f32, _> = x_train_frame.to_ndarray::<Float32Type>().unwrap();
    let x_test_array: Array<f32, _> = x_test_frame.to_ndarray::<Float32Type>().unwrap();

    // println!("{:?}", x_train_frame);
    // println!("{:?}", x_train_frame.transpose());

    (x_train_array, x_test_array, y_train_array, y_test_array)
}
