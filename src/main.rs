extern crate xgboost;

use labello::*;

use xgboost::{parameters, Booster, DMatrix};

use polars::df;
use polars::prelude::*;

fn main() {
    println!("Hello, world!");

    let x: DataFrame = CsvReader::from_path("datasets/urban/train_data.csv")
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();

    let y = x.column("class").unwrap().to_owned();

    let z: Vec<_> = y.utf8().into_iter().collect();

    for (i, elem) in y.utf8().into_iter().enumerate() {
        println!("i: {:?}, {:?}", i, elem);
    }

    // println!("{:?}", z);
    //
    // let enctype = EncoderType::Ordinal;
    // let config = Config{
    //     mapping_function:None,
    //     max_nclasses: Some(y.n_unique().unwrap() as u64),
    // };
    // let mut enc: Encoder<String> = Encoder::new(Some(enctype));
    //
    // enc.fit(&z, &config);
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //

    println!("{:?}", x);
}
