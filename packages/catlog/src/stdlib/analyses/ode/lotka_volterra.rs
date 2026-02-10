//! Lotka-Volterra ODE analysis of models.
//!
//! The main entry point for this module is
//! [`lotka_volterra_analysis`](SignedCoefficientBuilder::lotka_volterra_analysis).

use std::collections::HashMap;

use indexmap::IndexMap;
use nalgebra::{DMatrix, DVector};
use num_traits::{One, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde-wasm")]
use tsify::Tsify;

use super::{ODEAnalysis, SignedCoefficientBuilder};
use crate::dbl::model::{DiscreteDblModel, FgDblModel, ModalDblModel, MutDblModel};
use crate::dbl::theory::{ModalMorType, ModalObType};
use crate::simulate::ode::{NumericalPolynomialSystem, ODEProblem, ODESystem, PolynomialSystem};
use crate::zero::alg::Polynomial;
use crate::zero::rig::Monomial;
use crate::zero::{NameSegment, name};
use crate::{one::QualifiedPath, zero::QualifiedName};

/// A Lotka-Volterra dynamical system.
///
/// A system of ODEs that is affine in its *logarithmic* derivative. These are
/// sometimes called the "generalized Lotka-Volterra equations." For more, see
/// [Wikipedia](https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation).
#[derive(Clone, Debug, PartialEq)]
pub struct LotkaVolterraSystem {
    interaction_coeffs: DMatrix<f32>,
    growth_rates: DVector<f32>,
}

impl LotkaVolterraSystem {
    /// Constructs a new Lokta-Volterra system with the given parameters.
    pub fn new(A: DMatrix<f32>, b: DVector<f32>) -> Self {
        Self { interaction_coeffs: A, growth_rates: b }
    }

    /// Converts to a numerical polynomial system.
    pub fn to_polynomial(self) -> NumericalPolynomialSystem<u8> {
        NumericalPolynomialSystem {
            components: self
                .interaction_coeffs
                .row_iter()
                .enumerate()
                .zip(self.growth_rates.into_iter())
                .map(|((i, row), rate)| {
                    Polynomial::<_, f32, _>::generator(i)
                        * (row
                            .iter()
                            .enumerate()
                            .map(|(j, a)| Polynomial::generator(j) * *a)
                            .sum::<Polynomial<_, _, _>>()
                            + *rate)
                })
                .collect(),
        }
    }
}

impl ODESystem for LotkaVolterraSystem {
    fn vector_field(&self, dx: &mut DVector<f32>, x: &DVector<f32>, _t: f32) {
        let A = &self.interaction_coeffs;
        let b = &self.growth_rates;
        *dx = (A * x + b).component_mul(x);
    }
}

/// Lotka-Volterra ODE analysis for ecological models.
pub struct LotkaVolterraAnalysis {
    /// Object type for species/populations
    pub species_ob_type: ModalObType,
    /// Morphism type for interactions between species
    pub interaction_mor_type: ModalMorType,
}

impl Default for LotkaVolterraAnalysis {
    fn default() -> Self {
        let ob_type = ModalObType::new(name("Species"));
        Self {
            species_ob_type: ob_type.clone(),
            interaction_mor_type: ModalMorType::Zero(ob_type),
        }
    }
}

/// Symbolic parameter in Lotka-Volterra polynomial system.
type Parameter<Id> = Polynomial<Id, f32, i8>;

impl LotkaVolterraAnalysis {
    /// Creates a Lotka-Volterra system with symbolic interaction coefficients and growth rates.
    ///
    /// The resulting system represents:
    /// dx_i/dt = x_i * (r_i + sum_j(a_ij * x_j))
    ///
    /// where r_i are growth rates and a_ij are interaction coefficients.
    pub fn build_system(
        &self,
        model: &ModalDblModel,
    ) -> PolynomialSystem<QualifiedName, Parameter<QualifiedName>, i8> {
        let mut sys = PolynomialSystem::new();

        // Collect all species for building the complete system
        let species: Vec<_> = model.ob_generators_with_type(&self.species_ob_type).collect();

        for ob in species.iter() {
            // Build the interior polynomial: growth_rate + sum(interaction_coefficients * other_species)
            let mut interior: Polynomial<QualifiedName, Parameter<QualifiedName>, i8> =
                Polynomial::zero();

            // Add growth rate term
            let growth_rate_name = ob.snoc(name("growth").only().unwrap());
            let growth_param = Parameter::generator(growth_rate_name);
            let growth_term: Polynomial<QualifiedName, Parameter<QualifiedName>, i8> =
                [(growth_param, Monomial::one())].into_iter().collect();
            interior = interior + growth_term;

            // Add interaction terms
            for mor in model.mor_generators_with_type(&self.interaction_mor_type) {
                let src = MutDblModel::get_dom(model, &mor)
                    .and_then(|ob| ob.clone().unwrap_generator().into())
                    .expect("Interaction morphism should have single species as domain");
                let tgt = MutDblModel::get_cod(model, &mor)
                    .and_then(|ob| ob.clone().unwrap_generator().into())
                    .expect("Interaction morphism should have single species as codomain");

                // Only add this interaction if it affects the current species (src == ob)
                if src == *ob {
                    // Create the interaction coefficient as a symbolic parameter
                    let param = Parameter::generator(mor);

                    // Create monomial representing the target species population
                    let mon: Monomial<QualifiedName, i8> = [(tgt, 1)].into_iter().collect();

                    // Create the polynomial term: coefficient * target_population
                    let interaction_term: Polynomial<QualifiedName, Parameter<QualifiedName>, i8> =
                        [(param, mon)].into_iter().collect();

                    interior = interior + interaction_term;
                }
            }

            // Multiply the interior by the species variable: x_i * (interior)
            let species_var: Polynomial<QualifiedName, Parameter<QualifiedName>, i8> =
                Polynomial::generator(ob.clone());
            let full_term = species_var * interior;

            sys.add_term(ob.clone(), full_term);
        }

        sys
    }
}

/// Substitutes numerical coefficients into a symbolic Lotka-Volterra system.
pub fn extend_lotka_volterra_scalars(
    sys: PolynomialSystem<QualifiedName, Parameter<QualifiedName>, i8>,
    data: &LotkaVolterraProblemData,
) -> PolynomialSystem<QualifiedName, f32, i8> {
    sys.extend_scalars(|poly| {
        poly.eval(|param_name| {
            // Check if it's an interaction coefficient
            if let Some(&coeff) = data.interaction_coeffs.get(param_name) {
                return coeff;
            }
            // Check if it's a growth rate parameter
            // Growth rate parameters have the form species.growth
            if let Some(growth_seg) = param_name.segments().last() {
                if matches!(growth_seg, NameSegment::Text(text) if text.as_str() == "growth") {
                    // The species name is all segments except the last
                    let segments: Vec<_> = param_name.segments().cloned().collect();
                    if segments.len() > 1 {
                        let species_id =
                            QualifiedName::from(segments[..segments.len() - 1].to_vec());
                        return data.growth_rates.get(&species_id).copied().unwrap_or_default();
                    }
                }
            }
            // Default to 0 if parameter not found
            0.0
        })
    })
}

/// Builds the numerical ODE analysis for a Lotka-Volterra system whose scalars have been substituted.
pub fn into_lotka_volterra_analysis(
    sys: PolynomialSystem<QualifiedName, f32, i8>,
    data: LotkaVolterraProblemData,
) -> ODEAnalysis<NumericalPolynomialSystem<i8>> {
    let ob_index: IndexMap<_, _> =
        sys.components.keys().cloned().enumerate().map(|(i, x)| (x, i)).collect();
    let n = ob_index.len();

    let initial_values = ob_index
        .keys()
        .map(|ob| data.initial_values.get(ob).copied().unwrap_or_default());
    let x0 = DVector::from_iterator(n, initial_values);

    let num_sys = sys.to_numerical();
    let problem = ODEProblem::new(num_sys, x0).end_time(data.duration);

    ODEAnalysis::new(problem, ob_index)
}

/// Data defining a Lotka-Volterra ODE problem for a model.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde-wasm", derive(Tsify))]
#[cfg_attr(
    feature = "serde-wasm",
    tsify(into_wasm_abi, from_wasm_abi, hashmap_as_object)
)]
pub struct LotkaVolterraProblemData {
    /// Map from morphism IDs to interaction coefficients (arbitrary reals).
    #[cfg_attr(feature = "serde", serde(rename = "interactionCoefficients"))]
    pub interaction_coeffs: HashMap<QualifiedName, f32>,

    /// Map from object IDs to growth rates (arbitrary real numbers).
    #[cfg_attr(feature = "serde", serde(rename = "growthRates"))]
    pub growth_rates: HashMap<QualifiedName, f32>,

    /// Map from object IDs to initial values (nonnegative reals).
    #[cfg_attr(feature = "serde", serde(rename = "initialValues"))]
    pub initial_values: HashMap<QualifiedName, f32>,

    /// Duration of simulation.
    pub duration: f32,
}

impl SignedCoefficientBuilder<QualifiedName, QualifiedPath> {
    /// Lotka-Volterra ODE analysis for a model of a double theory.
    ///
    /// The main application we have in mind is the Lotka-Volterra ODE semantics for
    /// signed graphs described in our [paper on regulatory
    /// networks](crate::refs::RegNets).
    pub fn lotka_volterra_analysis(
        &self,
        model: &DiscreteDblModel,
        data: LotkaVolterraProblemData,
    ) -> ODEAnalysis<NumericalPolynomialSystem<u8>> {
        let (matrix, ob_index) = self.build_matrix(model, &data.interaction_coeffs);
        let n = ob_index.len();

        let growth_rates =
            ob_index.keys().map(|ob| data.growth_rates.get(ob).copied().unwrap_or_default());
        let b = DVector::from_iterator(n, growth_rates);

        let initial_values = ob_index
            .keys()
            .map(|ob| data.initial_values.get(ob).copied().unwrap_or_default());
        let x0 = DVector::from_iterator(n, initial_values);

        let system = LotkaVolterraSystem::new(matrix, b).to_polynomial();
        let problem = ODEProblem::new(system, x0).end_time(data.duration);
        ODEAnalysis::new(problem, ob_index)
    }
}

#[cfg(test)]
mod test {
    use std::rc::Rc;

    use super::*;
    use crate::dbl::theory::ModalDblTheory;
    use crate::simulate::ode::textplot_ode_result;
    use crate::stdlib;
    use crate::{one::Path, zero::name};

    #[test]
    fn symbolic_predator_prey_system() {
        use crate::dbl::modal::model::ModalOb;
        use crate::dbl::theory::ModalDblTheory;
        use expect_test::expect;

        // Create a simple Lotka-Volterra model with two species
        let th = Rc::new(ModalDblTheory::new());
        let mut model = ModalDblModel::new(th);

        let analysis = LotkaVolterraAnalysis::default();

        // Create two species
        let x = name("x");
        let y = name("y");
        model.add_ob(x.clone(), analysis.species_ob_type.clone());
        model.add_ob(y.clone(), analysis.species_ob_type.clone());

        // Create interactions
        let a = name("a");
        let b = name("b");

        model.add_mor(
            a.clone(),
            ModalOb::Generator(y.clone()),
            ModalOb::Generator(x.clone()),
            analysis.interaction_mor_type.clone(),
        );
        model.add_mor(
            b.clone(),
            ModalOb::Generator(x.clone()),
            ModalOb::Generator(y.clone()),
            analysis.interaction_mor_type.clone(),
        );

        // Build the symbolic system
        let sym_sys = analysis.build_system(&model);

        // Verify the symbolic system structure
        let expected = expect![[r#"
            dx = (x.growth) x + b x y
            dy = a x y + (y.growth) y
        "#]];
        expected.assert_eq(&sym_sys.to_string());

        // Create problem data
        let data = LotkaVolterraProblemData {
            interaction_coeffs: [(b, -1.0), (a, 1.0)].into_iter().collect(),
            growth_rates: [(x.clone(), 2.0), (y.clone(), -1.0)].into_iter().collect(),
            initial_values: [(x.clone(), 1.0), (y.clone(), 1.0)].into_iter().collect(),
            duration: 10.0,
        };

        // Substitute numerical values
        let num_sys = extend_lotka_volterra_scalars(sym_sys, &data);

        // Create the analysis
        let ode_analysis = into_lotka_volterra_analysis(num_sys, data);

        // Verify the ODE problem was created correctly
        assert_eq!(ode_analysis.problem.initial_values.len(), 2);
        assert_eq!(ode_analysis.problem.end_time, 10.0);

        // Solve the ODE to verify it works
        let result = ode_analysis.problem.solve_rk4(0.1).unwrap();
        let expected = expect![[r#"
                ⡁⠀⠀⠀⠀⠀⠀⠀⢠⠊⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 3.5
                ⠄⠀⠀⠀⠀⠀⠀⠀⡇⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠂⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⡁⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠄⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠂⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⡁⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠄⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠂⠀⠀⠀⠀⣸⡀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⡁⠀⠀⠀⡎⡜⢣⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⢹⠸⡀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠄⠀⠀⡸⠀⡇⠈⡆⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⡜⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠄
                ⠂⠀⢠⠃⢸⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⡎⢀⠇⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀
                ⡁⠀⡎⠀⡎⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⡸⠀⡸⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⢠⠃⠀
                ⠄⢰⠁⢰⠁⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⢀⠇⢀⠇⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⡎⠀⠀
                ⢂⠇⢀⠇⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⡜⠀⡜⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀⢰⠁⡰⠁
                ⡝⡠⠊⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⣀⢰⣁⠜⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⣀⢀⢇⡰⠁⠀
                ⠍⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⠁⠀⠀⠀
                ⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀
                ⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀
                ⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⡀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⡀⠀⠀⢀⡠⠔⠁⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 0.4
                0.0                                           10.0
        "#]];
        expected.assert_eq(&textplot_ode_result(&ode_analysis.problem, &result));
    }

    fn create_predator_prey() -> ODEProblem<NumericalPolynomialSystem<u8>> {
        let A = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let b = DVector::from_column_slice(&[2.0, -1.0]);
        let sys = LotkaVolterraSystem::new(A, b).to_polynomial();

        let x0 = DVector::from_column_slice(&[1.0, 1.0]);
        ODEProblem::new(sys, x0).end_time(10.0)
    }

    #[test]
    fn predator_prey() {
        let th = Rc::new(stdlib::theories::th_signed_category());
        let neg_feedback = stdlib::models::negative_feedback(th);

        let data = LotkaVolterraProblemData {
            interaction_coeffs: [(name("positive"), 1.0), (name("negative"), 1.0)]
                .into_iter()
                .collect(),
            growth_rates: [(name("x"), 2.0), (name("y"), -1.0)].into_iter().collect(),
            initial_values: [(name("x"), 1.0), (name("y"), 1.0)].into_iter().collect(),
            duration: 10.0,
        };
        let analysis = SignedCoefficientBuilder::new(name("Object"))
            .add_positive(Path::Id(name("Object")))
            .add_negative(Path::single(name("Negative")))
            .lotka_volterra_analysis(&neg_feedback, data);
        assert_eq!(analysis.problem, create_predator_prey());
    }
}
