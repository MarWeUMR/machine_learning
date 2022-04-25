use ml::qs::QuickArgSort;

mod ml;

fn main() {
    println!("==========================");
    println!("XG with Landcover");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasts::Landcover);
    println!("--------------------------\n\n");
    
    println!("==========================");
    println!("XG with Titanic");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasts::Titanic);
    println!("--------------------------\n\n");
    
    println!("==========================");
    println!("XG with Urban");
    println!("==========================\n");
    ml::xgbindings::run(ml::xgbindings::Datasts::Urban);
    println!("--------------------------\n\n");

    println!("==========================");
    println!("SMARTCORE with Titanic");
    println!("==========================\n");
    ml::smrtcore::run(ml::smrtcore::Datasets::Titanic);
    println!("--------------------------\n\n");
    
    println!("==========================");
    println!("SMARTCORE with Urban");
    println!("==========================\n");
    ml::smrtcore::run(ml::smrtcore::Datasets::Urban);
    println!("--------------------------\n\n");
    
    println!("==========================");
    println!("SMARTCORE with Landcover");
    println!("==========================\n");
    ml::smrtcore::run(ml::smrtcore::Datasets::Landcover);
    println!("--------------------------\n\n");
}
