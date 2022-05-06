mod ml;

// TODOOOOOO
// return trype für split_data ist bei den arrays ein problem, da momentan nur float arrays rausgegeben werden.
// sollte hier für verschiedene typen gearbeitet werden, oder alles lieber enkodieren?


fn main() {
    println!("XGBOOST\n");
    // ml::xgbindings::run(ml::xgbindings::Datasets::Landcover);
    println!("--------------------------\n\n");

    // ml::xgbindings::run(ml::xgbindings::Datasets::Titanic);

    ml::xgbindings::run(ml::xgbindings::Datasets::Urban);

    // ml::xgbindings::run(ml::xgbindings::Datasets::Boston);

    // ml::xgbindings::run(ml::xgbindings::Datasets::Cancer);

    // ml::xgbindings::run(ml::xgbindings::Datasets::Iris);

    // println!("==========================");
    // println!("SMARTCORE with Titanic");
    // println!("==========================\n");
    // // vvv --- Dataset bug??
    // // ml::smrtcore::run(ml::smrtcore::Datasets::Titanic);
    // println!("--------------------------\n\n");
    //
    // println!("==========================");
    // println!("SMARTCORE with Urban");
    // println!("==========================\n");
    // // ml::smrtcore::run(ml::smrtcore::Datasets::Urban);
    // println!("--------------------------\n\n");
    //
    // println!("==========================");
    // println!("SMARTCORE with Landcover");
    // println!("==========================\n");
    // // vvvv----- dauert ewig und kein killswitch
    // // ml::smrtcore::run(ml::smrtcore::Datasets::Landcover);
    // println!("--------------------------\n\n");
    //
    // println!("==========================");
    // println!("SMARTCORE with Boston Housing");
    // println!("==========================\n");
    // // ml::smrtcore::run(ml::smrtcore::Datasets::Boston);
    // println!("--------------------------\n\n");
    //
    // println!("==========================");
    // println!("SMARTCORE with Cancer");
    // println!("==========================\n");
    // // ml::smrtcore::run(ml::smrtcore::Datasets::Cancer);
    // println!("--------------------------\n\n");
    //
    // println!("TANGRAM\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Boston);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Iris);
    // println!("\n\n--------------------------\n\n");
    // // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Cancer);
    // println!("\n\n--------------------------\n\n");
    // // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Titanic);
    // println!("\n\n--------------------------\n\n");
    // // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Urban);
    // println!("\n\n--------------------------\n\n");
    // ml::tangram_wrapper::run(ml::tangram_wrapper::Datasets::Landcover);
}
