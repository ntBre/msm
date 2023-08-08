use std::fs::read_to_string;

use super::*;

#[test]
fn load() {
    let s = read_to_string("testfiles/first.json").unwrap();
    // Ligand silently gets broadcast into a list of Ligands, which might be
    // the essential problem
    let ligand = run(s);
    dbg!(ligand);
}
