use std::{f64::consts::PI, fs::File, io::Write};

use graph::Graph;
use ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2,
};
use ndarray_linalg::{Eig, Norm};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(test)]
mod tests;

mod graph;

const KCAL_TO_KJ: f64 = 4.184;

/// Degrees to radians
const DEG_TO_RAD: f64 = PI / 180.0;

type Dvec = Array1<f64>;

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct HarmonicBondParameter {
    length: f64,
    k: f64,
    #[serde(default)]
    atoms: (usize, usize),
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

    fn create_parameter(&mut self, atoms: (usize, usize), length: f64, k: f64) {
        if self.parameters.is_none() {
            self.parameters = Some(Vec::new());
        }

        // always delete old params for some reason
        self.remove_parameter(atoms);
        self.parameters
            .as_mut()
            .unwrap()
            .push(HarmonicBondParameter { length, k, atoms });
    }

    fn remove_parameter(&mut self, atoms: (usize, usize)) {
        let Some(parameter) = self.get(atoms) else {
            return;
        };
        let idx = self
            .parameters
            .as_ref()
            .unwrap()
            .iter()
            .position(|p| p == parameter)
            .unwrap();
        let parameters = self.parameters.as_mut().unwrap();
        parameters.remove(idx);
    }

    fn get(&self, index: (usize, usize)) -> Option<&HarmonicBondParameter> {
        self.parameters.as_ref().unwrap().iter().find(|&parameter| {
            parameter.atoms == index || parameter.atoms == (index.1, index.0)
        })
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

    fn create_parameter(
        &mut self,
        _angle: (usize, usize, usize),
        _deg_to_rad: f64,
        _conversion: f64,
    ) {
        todo!()
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
pub struct Ligand {
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

    /// unwrap self.coordinates and return the whole vector
    fn coordinates(&self) -> &Vec<f64> {
        self.coordinates.as_ref().unwrap()
    }

    /// unwrap self.coordinates and return the ith entry
    fn coordinates_at(&self, i: usize) -> Array1<f64> {
        coordinates_at(self.coordinates(), i)
    }

    fn to_topology(&self) -> Graph {
        let mut graph = Graph::new();
        for atom in &self.atoms {
            graph.add_node(atom.atom_index);
        }

        for bond in self.bonds() {
            graph.add_edge(bond.atom1_index, bond.atom2_index);
        }
        graph
    }

    pub(crate) fn bonds(&self) -> &Vec<Bond> {
        self.bonds.as_ref().unwrap()
    }

    fn angles(&self) -> Vec<(usize, usize, usize)> {
        let mut angles = Vec::new();
        let topology = self.to_topology();
        for node in topology.nodes() {
            let mut bonded: Vec<_> = topology.neighbors(node).collect();
            bonded.sort();

            // check that the atom has more than one bond
            if bonded.len() < 2 {
                continue;
            }

            // find all possible angle combinations from the list
            for i in 0..bonded.len() {
                for j in i + 1..bonded.len() {
                    angles.push((*bonded[i], *node, *bonded[j]));
                }
            }
        }

        angles
    }
}

fn coordinates_at(v: &[f64], i: usize) -> Array1<f64> {
    Array1::from(v.chunks_exact(3).nth(i).unwrap().to_vec())
}

struct ModSemMaths;

impl ModSemMaths {
    fn force_constant_bond(
        bond: &(usize, usize),
        eigenvals: &Array3<Complex64>,
        eigenvecs: &Array4<Complex64>,
        coordinates: &[f64],
    ) -> f64 {
        let (atom_a, atom_b) = *bond;
        let eigenvals_ab = eigenvals.slice(s![atom_a, atom_b, ..]);
        let eigenvecs_ab = eigenvecs.slice(s![.., .., atom_a, atom_b]);

        let unit_vectors_ab = Array1::from_iter(
            ModSemMaths::unit_vector_along_bond(coordinates, *bond)
                .into_iter()
                .map(|r| Complex64::new(r, 0.0)),
        );

        let mut sum = 0.0;
        for i in 0..3 {
            let p = eigenvals_ab[i]
                * unit_vectors_ab.dot(&eigenvecs_ab.slice(s![.., i]));
            // not really sure what else to do here, but I think this is always
            // true
            assert_eq!(p.im, 0.0);
            sum += p.re.abs();
        }

        -0.5 * sum
    }

    fn unit_vector_along_bond(
        coordinates: &[f64],
        (atom_a, atom_b): (usize, usize),
    ) -> Array1<f64> {
        let diff_ab = coordinates_at(coordinates, atom_b)
            - coordinates_at(coordinates, atom_a);

        &diff_ab / diff_ab.norm()
    }

    /// returns the vector in plane ABC and perpendicular to AB
    fn u_pa_from_angles(
        (a, b, c): (usize, usize, usize),
        coordinates: &[f64],
    ) -> Array1<f64> {
        let u_ab = ModSemMaths::unit_vector_along_bond(coordinates, (a, b));
        let u_cb = ModSemMaths::unit_vector_along_bond(coordinates, (c, b));

        let u_n = ModSemMaths::unit_vector_normal_to_bond(&u_cb, &u_ab);

        ModSemMaths::unit_vector_normal_to_bond(&u_n, &u_ab)
    }

    fn force_constant_angle(
        angle: (usize, usize, usize),
        bond_lens: &Array2<f64>,
        eigenvals: &Array3<Complex64>,
        eigenvecs: &Array4<Complex64>,
        coords: &[f64],
        scalings: &[f64],
    ) -> (f64, f64) {
        let (atom_a, atom_b, atom_c) = angle;

        let u_ab =
            ModSemMaths::unit_vector_along_bond(coords, (atom_a, atom_b));
        let u_cb =
            ModSemMaths::unit_vector_along_bond(coords, (atom_c, atom_b));

        let bond_len_ab = bond_lens[(atom_a, atom_b)];
        let eigenvals_ab = eigenvals.slice(s![atom_a, atom_b, ..]);
        let eigenvecs_ab = eigenvecs.slice(s![..3, ..3, atom_a, atom_b]);

        let bond_len_bc = bond_lens[(atom_b, atom_c)];
        let eigenvals_cb = eigenvals.slice(s![atom_c, atom_b, ..]);
        let eigenvecs_cb = eigenvecs.slice(s![..3, ..3, atom_c, atom_b]);

        // Normal vector to angle plane found
        let u_n = ModSemMaths::unit_vector_normal_to_bond(&u_cb, &u_ab);

        // Angle is linear:
        let abs = (&u_cb - &u_ab).norm().abs();

        let (mut k_theta, theta_0);
        if abs < 0.01 || (1.99 < abs && abs < 2.01) {
            // Scalings are set to 1.
            (k_theta, theta_0) = ModSemMaths::f_c_a_special_case(
                u_ab,
                u_cb,
                [bond_len_ab, bond_len_bc],
                [eigenvals_ab, eigenvals_cb],
                [eigenvecs_ab, eigenvecs_cb],
            );
        } else {
            let u_pa = ModSemMaths::unit_vector_normal_to_bond(&u_n, &u_ab);
            let u_pc = ModSemMaths::unit_vector_normal_to_bond(&u_cb, &u_n);

            // Scaling due to additional angles - Modified Seminario Part
            let mut sum = 0.0;
            for i in 0..3 {
                let p = eigenvals_ab[i]
                    * ModSemMaths::dot_product(
                        &u_pa,
                        eigenvecs_ab.slice(s![.., i]),
                    );
                sum += p.re.abs();
            }
            let sum_first = sum / scalings[0];

            let mut sum = 0.0;
            for i in 0..3 {
                let p = eigenvals_cb[i]
                    * ModSemMaths::dot_product(
                        &u_pc,
                        eigenvecs_cb.slice(s![.., i]),
                    );
                sum += p.re.abs();
            }
            let sum_second = sum / scalings[1];

            // Added as two springs in series
            k_theta = (1.0 / ((bond_len_ab * bond_len_ab) * sum_first))
                + (1.0 / ((bond_len_bc * bond_len_bc) * sum_second));
            k_theta = 1.0 / k_theta;

            // Change to OPLS form
            k_theta = (k_theta * 0.5).abs();

            // Equilibrium Angle
            theta_0 = u_ab.dot(&u_cb).acos().to_degrees();
        }

        (k_theta, theta_0)
    }

    fn unit_vector_normal_to_bond(b: &Dvec, c: &Dvec) -> Dvec {
        assert_eq!(b.len(), 3);
        assert_eq!(c.len(), 3);

        // unfortunately ndarray doesn't seem to have a cross product
        let cross = Dvec::from(vec![
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ]);

        &cross / cross.norm()
    }

    fn f_c_a_special_case(
        _u_ab: Dvec,
        _u_cb: Dvec,
        _bond_len_bc: [f64; 2],
        _eigenvals_cb: [ArrayView1<Complex64>; 2],
        _eigenvecs_cb: [ArrayView2<Complex64>; 2],
    ) -> (f64, f64) {
        todo!()
    }

    fn dot_product(
        _u_pa: &Dvec,
        _s: ndarray::ArrayBase<
            ndarray::ViewRepr<&num_complex::Complex<f64>>,
            ndarray::Dim<[usize; 1]>,
        >,
    ) -> Complex64 {
        todo!()
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
                let ci = molecule.coordinates_at(i);
                let cj = molecule.coordinates_at(j);
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
        let bonds = molecule.to_topology().edges();
        const CONVERSION: f64 = 200.0 * KCAL_TO_KJ;

        let mut k_b = vec![0.0; bonds.len()];
        let mut bond_len_list = vec![0.0; bonds.len()];

        // TODO I DO NOT want to do this long term. please let me just return
        // what I need. It doesn't appear to be used by my Python script, so
        // hopefully I can just use it temporarily for debugging and then
        // delete.
        let mut bond_file =
            File::create("Modified_Seminario_Bonds.txt").unwrap();

        for (pos, bond) in bonds.iter().enumerate() {
            let ab = ModSemMaths::force_constant_bond(
                bond,
                eigenvals,
                eigenvecs,
                molecule.coordinates(),
            );

            let ba = ModSemMaths::force_constant_bond(
                &(bond.1, bond.0), // bond[::-1]
                eigenvals,
                eigenvecs,
                molecule.coordinates(),
            );

            // TODO may need real part of the first term
            k_b[pos] = ((ab + ba) / 2.0)
                * (self.vibrational_scaling * self.vibrational_scaling);

            bond_len_list[pos] = bond_lens[*bond];

            write!(
                bond_file,
                "{}-{}  ",
                molecule.atoms[bond.0].atom_name.as_ref().unwrap(),
                molecule.atoms[bond.1].atom_name.as_ref().unwrap(),
            )
            .unwrap();
            writeln!(
                bond_file,
                "{}    {}    {}    {}",
                k_b[pos], bond_len_list[pos], bond.0, bond.1,
            )
            .unwrap();

            molecule.bond_force.create_parameter(
                *bond,
                bond_len_list[pos] / 10.0,
                CONVERSION * k_b[pos],
            );
        }
    }

    fn calculate_angles(
        &self,
        eigenvals: &Array3<Complex64>,
        eigenvecs: &Array4<Complex64>,
        molecule: &mut Ligand,
        bond_lens: &Array2<f64>,
    ) {
        // A structure is created with the index giving the central atom of the
        // angle; an array then lists the angles with that central atom. e.g.
        // central_atoms_angles[3] contains an array of angles with central atom
        // 3.

        // connectivity information for MSM
        let mut central_atoms_angles = Vec::new();
        for coord in 0..molecule.atoms.len() {
            central_atoms_angles.push(Vec::new());
            for (count, angle) in molecule.angles().iter().enumerate() {
                if coord == angle.1 {
                    // for angle ABC atoms A and C and C and A are written to array
                    central_atoms_angles[coord].push([angle.0, angle.2, count]);
                    central_atoms_angles[coord].push([angle.2, angle.0, count]);
                }
            }
        }

        // sort rows by atom number
        for coord in central_atoms_angles.iter_mut() {
            coord.sort_by_key(|x| x[0]);
        }

        // find normals u_pa for each angle
        let mut unit_pa_all_angles = Vec::new();
        for i in 0..central_atoms_angles.len() {
            unit_pa_all_angles.push(Vec::new());
            for j in 0..central_atoms_angles[i].len() {
                // for the angle at central_atoms_angles[i][j,:], the u_pa value
                // is found for the plane ABC and bond AB, where ABC corresponds
                // to the order of the arguments. This is why the reverse order
                // is also added.
                let angle = (
                    central_atoms_angles[i][j][0],
                    i,
                    central_atoms_angles[i][j][1],
                );
                unit_pa_all_angles[i].push(ModSemMaths::u_pa_from_angles(
                    angle,
                    molecule.coordinates(),
                ));
            }
        }

        // finds the contributing factors from the other angle terms
        let mut scaling_factor_all_angles = Vec::new();
        for i in 0..central_atoms_angles.len() {
            scaling_factor_all_angles.push(Vec::new());
            for j in 0..central_atoms_angles[i].len() {
                let mut n = 1;
                let mut m = 1;
                let mut extra_contribs = 0.0;
                scaling_factor_all_angles[i].push((0.0, 0));

                // position in angle list
                scaling_factor_all_angles[i][j].1 =
                    central_atoms_angles[i][j][2];

                // goes through the list of angles with the same central atom,
                // then computes the term needed for MSM.

                // forward direction, finds the same bonds with the central atom
                // i
                while ((j + n) < central_atoms_angles[i].len())
                    && central_atoms_angles[i][j][0]
                        == central_atoms_angles[i][j + n][0]
                {
                    // TODO can I construct some kind of view instead?
                    let v1 = Array1::from(unit_pa_all_angles[i][j].clone());
                    let v2 = Array1::from(unit_pa_all_angles[i][j + n].clone());
                    extra_contribs += v1.dot(&v2).abs().powi(2);
                    n += 1;
                }

                // backward direction, finds the same bonds with central atom i
                while (j >= m)
                    && central_atoms_angles[i][j][0]
                        == central_atoms_angles[i][j - m][0]
                {
                    let v1 = Array1::from(unit_pa_all_angles[i][j].clone());
                    let v2 = Array1::from(unit_pa_all_angles[i][j - m].clone());
                    extra_contribs += v1.dot(&v2).abs().powi(2);
                    m += 1;
                }

                scaling_factor_all_angles[i][j].0 = 1.0;
                if n != 1 || m != 1 {
                    // finds the mean value of the additional contribution
                    scaling_factor_all_angles[i][j].0 +=
                        extra_contribs / (m + n - 2) as f64;
                }
            }
        }

        let mut scaling_factors_angles_list = Vec::new();
        for _ in 0..molecule.angles().len() {
            scaling_factors_angles_list.push(Vec::new());
        }

        // orders the scaling factors according to the angle list
        for i in 0..central_atoms_angles.len() {
            for j in 0..central_atoms_angles[i].len() {
                scaling_factors_angles_list[scaling_factor_all_angles[i][j].1]
                    .push(scaling_factor_all_angles[i][j].0);
            }
        }

        let mut k_theta = Array1::zeros(molecule.angles().len());
        let mut theta_0 = Array1::zeros(molecule.angles().len());

        const CONVERSION: f64 = KCAL_TO_KJ * 2.0;

        let mut angle_file =
            File::create("Modified_Seminario_Angles.txt").unwrap();

        for (i, angle) in molecule.angles().iter().enumerate() {
            let scalings = &scaling_factors_angles_list[i];
            let mut rev_scalings = scalings.clone();
            rev_scalings.reverse();

            let (ab_k_theta, ab_theta_0) = ModSemMaths::force_constant_angle(
                *angle,
                bond_lens,
                eigenvals,
                eigenvecs,
                molecule.coordinates(),
                scalings,
            );
            let (ba_k_theta, ba_theta_0) = ModSemMaths::force_constant_angle(
                (angle.2, angle.1, angle.0),
                bond_lens,
                eigenvals,
                eigenvecs,
                molecule.coordinates(),
                &rev_scalings,
            );

            k_theta[i] = ((ab_k_theta + ba_k_theta) / 2.0)
                * (self.vibrational_scaling * self.vibrational_scaling);
            theta_0[i] = (ab_theta_0 + ba_theta_0) / 2.0;

            writeln!(
                angle_file,
                "{}-{}-{}  {:.3}   {:.3}   {}   {}   {}",
                molecule.atoms[angle.0].atom_name.as_ref().unwrap(),
                molecule.atoms[angle.1].atom_name.as_ref().unwrap(),
                molecule.atoms[angle.2].atom_name.as_ref().unwrap(),
                k_theta[i],
                theta_0[i],
                angle.0,
                angle.1,
                angle.2
            )
            .unwrap();

            molecule.angle_force.create_parameter(
                *angle,
                theta_0[i] * DEG_TO_RAD,
                k_theta[i] * CONVERSION,
            )
        }

        todo!()
    }
}

/// deserialize a vector of [Ligand]s from `s` and run the modified Seminario
/// method on them
pub fn run(s: String) -> Vec<Ligand> {
    let ligand: Vec<Ligand> = serde_json::from_str(&s).unwrap();
    let mod_sem = ModSeminario::new();
    mod_sem.run(ligand)
}
