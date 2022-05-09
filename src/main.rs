#![feature(unsize, coerce_unsized)]

mod ml;

// TODO
// heart disease datensatz testen
// missing values bei titanic?

fn main() {
    println!("XGBOOST\n");
    // ml::xgbindings::run(ml::xgbindings::Datasets::Landcover);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Titanic);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Urban);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Boston);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Cancer);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Iris);
    println!("--------------------------\n\n");

    println!("TANGRAM\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Boston);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Iris);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Cancer);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Titanic);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Urban);
    // println!("\n\n--------------------------\n\n");
    ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Landcover);
}
