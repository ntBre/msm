use std::io::{self, Read};

use ndarray::{s, Array, Array1, Array2, Array3, Array4};
use ndarray_linalg::{Eig, Norm};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
struct HarmonicBondParameter {
    length: f64,
    k: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct BondForce {
    #[serde(rename = "type")]
    typ: String,
    parameters: Option<Vec<HarmonicBondParameter>>,
}

impl BondForce {
    fn clear_parameters(&mut self) {
        if let Some(ref mut parameters) = &mut self.parameters {
            parameters.clear();
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct HarmonicAngleParameter {
    length: f64,
    k: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct AngleForce {
    #[serde(rename = "type")]
    typ: String,
    parameters: Option<Vec<HarmonicAngleParameter>>,
}

impl AngleForce {
    fn clear_parameters(&mut self) {
        if let Some(ref mut parameters) = &mut self.parameters {
            parameters.clear();
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Aim {
    volume: Option<()>,
    charge: Option<()>,
    c8: Option<()>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Atom {
    atomic_number: usize,
    atom_index: usize,
    atom_name: Option<String>,
    formal_charge: isize,
    aromatic: bool,
    stereochemistry: Option<String>,
    bonds: Option<Vec<usize>>,
    aim: Option<Aim>,
    dipole: Option<()>,
    quadrupole: Option<()>,
    cloud_pen: Option<()>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Bond {
    atom1_index: usize,
    atom2_index: usize,
    bond_order: f64,
    aromatic: bool,
    stereochemistry: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Ligand {
    atoms: Vec<Atom>,
    bonds: Option<Vec<Bond>>,
    coordinates: Option<Vec<f64>>,
    multiplicity: usize,
    name: String,
    provenance: Value,
    extra_sites: Value,
    #[serde(rename = "BondForce")]
    bond_force: BondForce,

    #[serde(rename = "AngleForce")]
    angle_force: AngleForce,

    #[serde(rename = "TorsionForce")]
    torsion_force: Value,

    #[serde(rename = "ImproperTorsionForce")]
    improper_torsion_force: Value,

    #[serde(rename = "RBTorsionForce")]
    rb_torsion_force: Value,

    #[serde(rename = "ImproperRBTorsionForce")]
    improper_rb_torsion_force: Value,

    #[serde(rename = "NonbondedForce")]
    non_bonded_force: BondForce,

    chargemol_coords: Value,

    hessian: Option<Vec<f64>>,

    qm_scans: Value,

    wbo: Value,
}

impl Ligand {
    fn hessian(&self) -> &Vec<f64> {
        self.hessian.as_ref().unwrap()
    }

    fn symmetrize_bonded_parameters(&mut self) {
        todo!()
    }

    fn coordinates(&self, i: usize) -> Array1<f64> {
        Array1::from(
            self.coordinates
                .as_ref()
                .unwrap()
                .chunks_exact(3)
                .nth(i)
                .unwrap()
                .to_vec(),
        )
    }
}

struct ModSeminario {
    vibrational_scaling: f64,
}

impl Default for ModSeminario {
    fn default() -> Self {
        Self {
            vibrational_scaling: 1.0,
        }
    }
}

impl ModSeminario {
    fn new() -> Self {
        Self::default()
    }

    fn run(&self, mut molecules: Vec<Ligand>) -> Vec<Ligand> {
        for molecule in molecules.iter_mut() {
            molecule.bond_force.clear_parameters();
            molecule.angle_force.clear_parameters();
            /// Hartrees to kilocalories per mole
            const HA_TO_KCAL_P_MOL: f64 = 627.509391;
            /// Bohrs to Angstroms
            const BOHR_TO_ANGS: f64 = 0.529177;
            const CONVERSION: f64 =
                HA_TO_KCAL_P_MOL / (BOHR_TO_ANGS * BOHR_TO_ANGS);

            let mut hessian = molecule.hessian().clone();
            for h in hessian.iter_mut() {
                *h *= CONVERSION;
            }

            self.modified_seminario_method(molecule, hessian);
            molecule.symmetrize_bonded_parameters();
        }
        molecules
    }

    fn modified_seminario_method(
        &self,
        molecule: &mut Ligand,
        hessian: Vec<f64>,
    ) {
        let size_mol = molecule.atoms.len();
        let mut eigenvecs: Array4<Complex64> =
            Array::zeros((3, 3, size_mol, size_mol));
        let mut eigenvals: Array3<Complex64> =
            Array::zeros((size_mol, size_mol, 3));
        let mut bond_lens = Array::zeros((size_mol, size_mol));
        let hessian =
            Array2::from_shape_vec((3 * size_mol, 3 * size_mol), hessian)
                .unwrap();

        for i in 0..size_mol {
            for j in 0..size_mol {
                // TODO use a dvector or whatever ndarray provides
                let ci = molecule.coordinates(i);
                let cj = molecule.coordinates(j);
                let diff_i_j = ci - cj;
                bond_lens[(i, j)] = diff_i_j.norm();

                let partial_hessian = hessian
                    .slice(s![(i * 3)..((i + 1) * 3), (j * 3)..(j + 1) * 3]);

                let (vals, vecs) = partial_hessian.eig().unwrap();
                vals.assign_to(eigenvals.slice_mut(s![i, j, ..]));
                vecs.assign_to(eigenvecs.slice_mut(s![.., .., i, j]));
            }
        }

        self.calculate_bonds(&eigenvals, &eigenvecs, molecule, &bond_lens);
        self.calculate_angles(&eigenvals, &eigenvecs, molecule, &bond_lens);
    }

    fn calculate_bonds(
        &self,
        eigenvals: &Array3<Complex64>,
        eigenvecs: &Array4<Complex64>,
        molecule: &mut Ligand,
        bond_lens: &Array2<f64>,
    ) {
        todo!()
    }

    fn calculate_angles(
        &self,
        eigenvals: &Array3<Complex64>,
        eigenvecs: &Array4<Complex64>,
        molecule: &mut Ligand,
        bond_lens: &Array2<f64>,
    ) {
        todo!()
    }
}

fn run(s: String) -> Vec<Ligand> {
    let ligand: Vec<Ligand> = serde_json::from_str(&s).unwrap();
    let mod_sem = ModSeminario::new();
    let ligand = mod_sem.run(ligand);
    ligand
}

fn main() {
    let mut buf = Vec::new();
    io::stdin().lock().read_to_end(&mut buf).unwrap();
    let s = String::from_utf8(buf).unwrap();
    let ligands = run(s);
    serde_json::to_string_pretty(&ligands).unwrap();
}

#[cfg(test)]
mod tests {
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
}
