use std::io::{self, Read};

use msm::run;

fn main() {
    let mut buf = Vec::new();
    io::stdin().lock().read_to_end(&mut buf).unwrap();
    let s = String::from_utf8(buf).unwrap();
    let ligands = run(s);
    serde_json::to_string_pretty(&ligands).unwrap();
}
