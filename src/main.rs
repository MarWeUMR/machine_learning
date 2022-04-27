mod ml;

fn main() {
    println!("==========================");
    println!("XG with Landcover");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasets::Landcover);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("XG with Titanic");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasets::Titanic);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("XG with Urban");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasets::Urban);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("XG with Boston");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasets::Boston);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("XG with Cancer");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasets::Cancer);
    println!("--------------------------\n\n");



    println!("==========================");
    println!("SMARTCORE with Titanic");
    println!("==========================\n");
    // vvv --- Dataset bug??
    // ml::smrtcore::run(ml::smrtcore::Datasets::Titanic);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("SMARTCORE with Urban");
    println!("==========================\n");
    ml::smrtcore::run(ml::smrtcore::Datasets::Urban);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("SMARTCORE with Landcover");
    println!("==========================\n");
    // vvvv----- dauert ewig und kein killswitch
    // ml::smrtcore::run(ml::smrtcore::Datasets::Landcover);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("SMARTCORE with Boston Housing");
    println!("==========================\n");
    // ml::smrtcore::run(ml::smrtcore::Datasets::Boston);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("SMARTCORE with Cancer");
    println!("==========================\n");
    // ml::smrtcore::run(ml::smrtcore::Datasets::Cancer);
    println!("--------------------------\n\n");
}
