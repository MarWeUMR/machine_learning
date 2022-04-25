use xgboost::{
    parameters::{self, learning::LearningTaskParametersBuilder, BoosterParametersBuilder},
    Booster, DMatrix,
};

use crate::ml::data_processing;

pub enum Datasts {
    Titanic,
    Landcover,
    Urban,
}

pub fn run(set: Datasts) {
    // specify dataset parameters
    let (dataset, target_column, xg_classifier) = get_target_column(set);

    // use python to preprocess data
    data_processing::run_through_python(dataset);

    // read preprocessed data to rust
    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_data_matrix(dataset, target_column);

    let train_shape = x_train_array.raw_dim();
    let test_shape = x_test_array.raw_dim();

    let num_rows_train = train_shape[0];
    let num_rows_test = test_shape[0];

    let mut x_train = DMatrix::from_dense(
        x_train_array
            .into_shape(train_shape[0] * train_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_train,
    )
    .unwrap();

    let mut x_test = DMatrix::from_dense(
        x_test_array
            .into_shape(test_shape[0] * test_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_test,
    )
    .unwrap();

    x_train
        .set_labels(y_train_array.as_slice().unwrap())
        .unwrap();

    x_test.set_labels(y_test_array.as_slice().unwrap()).unwrap();

    // optional
    // let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // RANDOM FOREST SETUP -----------------------------------

    // let rf_tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
    //     .subsample(0.8)
    //     .max_depth(5)
    //     .eta(1.0) // aka learning_rate
    //     .colsample_bytree(0.8)
    //     // .num_parallel_tree(100) // <- NOT AVAILABLE ?!
    //     .tree_method(parameters::tree::TreeMethod::Hist)
    //     .build()
    //     .unwrap();
    //
    // let rf_learning_params = LearningTaskParametersBuilder::default()
    //     .objective(xg_classifier)
    //     .build()
    //     .unwrap();
    //
    // let rf_booster_params = BoosterParametersBuilder::default()
    //     .booster_type(parameters::BoosterType::Tree(rf_tree_params))
    //     .learning_params(rf_learning_params)
    //     .build()
    //     .unwrap();
    //
    // let rf_params = parameters::TrainingParametersBuilder::default()
    //     .dtrain(&x_train)
    //     .boost_rounds(1)
    //     .booster_params(rf_booster_params)
    //     .build()
    //     .unwrap();
    //

    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&x_train)
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    // println!("{:?}", bst.predict(&dtest).unwrap());
    let preds = bst.predict(&x_test).unwrap();

    let labels = x_test.get_labels().unwrap();
    println!(
        "First 3 predicted labels: {} {} {}",
        labels[0], labels[1], labels[2]
    );

    // print error rate
    let num_correct: usize = preds.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).sum();
    println!(
        "error={} ({}/{} correct)",
        num_correct as f32 / preds.len() as f32,
        num_correct,
        preds.len()
    );
}

fn get_target_column<'a>(set: Datasts) -> (&'a str, &'a str, parameters::learning::Objective) {
    let result = match set {
        Datasts::Titanic => {
            println!("BinaryLogistic chosen");
            (
                "titanic",
                "Survived",
                parameters::learning::Objective::BinaryLogistic,
            )
        }
        Datasts::Urban => {
            println!("MultiSoftmax chosen");
            (
                "urban",
                "class",
                parameters::learning::Objective::MultiSoftmax(10),
            )
        }
        Datasts::Landcover => {
            println!("MultiSoftmax chosen");
            (
                "landcover",
                "Class_ID",
                parameters::learning::Objective::MultiSoftmax(10),
            )
        }
    };
    result
}
