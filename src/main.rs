
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;
use parquet::record::{RowAccessor};

use hnsw_rs::prelude::*;

use parquet::file::reader::{FileReader, SerializedFileReader};
use std::{fs::File, path::Path};


use std::io::{self, BufRead};


fn main() {
 
    let path = Path::new("./embedded_products_big.parquet.gzip");
    if let Ok(file) = File::open(&path) {
        let reader = SerializedFileReader::new(file).unwrap();
    let nb_elem = 100000000;
    
    let mut key_lookup: HashMap<String, usize> = HashMap::new();
    let mut value_lookup: HashMap<usize,String> = HashMap::new();
    
    let embeddings: Vec<Vec<f64>> =  reader.get_row_iter(None).unwrap().take(nb_elem).enumerate().map(|(index, row)| {
        let r = row.unwrap();
        let id = r.get_string(0).unwrap().clone();
        let embeddings: Vec<f64> = r.get_column_iter().skip(1).take(64).enumerate().map(|(x,_y)| r.get_double(x+1).unwrap()).collect();
        key_lookup.insert(id.clone(), index);
        value_lookup.insert(index,id);
        embeddings
    }).collect();

    let data_with_id: Vec<(&Vec<f64>, usize)> = embeddings.iter().zip(0..embeddings.len()).collect();

    let ef_c = 200;
    let max_nb_connection = 15;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let hns = Hnsw::<f64, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{});
    let mut start = ProcessTime::now();
    let mut begin_t = SystemTime::now();
    
    hns.parallel_insert(&data_with_id);
    let mut cpu_time: Duration =    start.elapsed();
    println!(" hnsw data insertion  cpu time {:?}", cpu_time); 
    println!(" hnsw data insertion parallel,   system time {:?} \n", begin_t.elapsed().unwrap()); 
    hns.dump_layer_info();
    println!(" parallel hnsw data nb point inserted {:?}", hns.get_nb_point());
    
    let mut line = String::new();
    let stdin = io::stdin();

    println!("ENTER PRODUCT ID");
    while let Ok(_) = stdin.lock().read_line(&mut line) {
        if let Some(index) = key_lookup.get(line.trim()) {
            start = ProcessTime::now();
            let embedding = embeddings.get(*index);
        
            let ef_search = max_nb_connection * 2;
            let knbn = 100;
        
            let neighbours = hns.search(&embedding.unwrap(), knbn, ef_search);
            cpu_time =    start.elapsed();
          
            for neighbour in neighbours.iter() {
               let id = value_lookup.get(&neighbour.d_id);
               println!("FOUND MATCH: {} (distance: {})", id.unwrap(), neighbour.distance);
                
            }

            println!("\n\nLookup took {}μs", cpu_time.as_micros());
            println!("ENTER PRODUCT ID");
        } else {
            println!("NOT FOUND!");
        }
        
    }  


    }
}
