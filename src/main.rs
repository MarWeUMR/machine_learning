mod ml;

fn main() {
    // ml::xgbindings::run(ml::xgbindings::Datasts::Landcover);
    // ml::xgbindings::run(ml::xgbindings::Datasts::Titanic);
    // ml::xgbindings::run(ml::xgbindings::Datasts::Urban);

    ml::smrtcore::run(ml::smrtcore::Datasets::Titanic);
}
