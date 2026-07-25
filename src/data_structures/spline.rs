//! Internal cubic B-spline primitives used by the validated KAN implementation.

/// The fixed polynomial degree used by the P0 KAN model.
pub(crate) const SPLINE_DEGREE: usize = 3;

#[derive(Clone, Debug)]
pub(crate) struct SplineActivation {
    pub(crate) coefficients: Vec<f32>,
    pub(crate) base_weight: f32,
    pub(crate) spline_weight: f32,
    grid_intervals: usize,
    domain: [f32; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct BasisEvaluation {
    pub(crate) values: Vec<f32>,
    pub(crate) derivatives: Vec<f32>,
}

impl SplineActivation {
    pub(crate) fn new(
        grid_intervals: usize,
        domain: [f32; 2],
        base_weight: f32,
        spline_weight: f32,
        coefficients: Vec<f32>,
    ) -> Self {
        Self {
            coefficients,
            base_weight,
            spline_weight,
            grid_intervals,
            domain,
        }
    }

    pub(crate) fn evaluate(&self, input: f32) -> (f32, BasisEvaluation) {
        let basis = self.basis(input);
        let spline = dot(&self.coefficients, &basis.values);
        (
            self.base_weight * silu(input) + self.spline_weight * spline,
            basis,
        )
    }

    pub(crate) fn spline_value_and_derivative(&self, basis: &BasisEvaluation) -> (f32, f32) {
        (
            dot(&self.coefficients, &basis.values),
            dot(&self.coefficients, &basis.derivatives),
        )
    }

    pub(crate) fn basis(&self, input: f32) -> BasisEvaluation {
        let knots = exterior_uniform_knots(self.grid_intervals, self.domain);
        let values = basis_values(&knots, SPLINE_DEGREE, input);
        let lower_values = basis_values(&knots, SPLINE_DEGREE - 1, input);
        let mut derivatives = vec![0.0; values.len()];

        for index in 0..values.len() {
            let left_denominator = knots[index + SPLINE_DEGREE] - knots[index];
            let right_denominator = knots[index + SPLINE_DEGREE + 1] - knots[index + 1];
            let left = if left_denominator == 0.0 {
                0.0
            } else {
                SPLINE_DEGREE as f32 * lower_values[index] / left_denominator
            };
            let right = if right_denominator == 0.0 {
                0.0
            } else {
                SPLINE_DEGREE as f32 * lower_values[index + 1] / right_denominator
            };
            derivatives[index] = left - right;
        }

        BasisEvaluation {
            values,
            derivatives,
        }
    }
}

pub(crate) fn coefficient_count(grid_intervals: usize) -> usize {
    grid_intervals + SPLINE_DEGREE
}

pub(crate) fn exterior_uniform_knots(grid_intervals: usize, domain: [f32; 2]) -> Vec<f32> {
    let interval_width = (domain[1] - domain[0]) / grid_intervals as f32;
    (0..(grid_intervals + 2 * SPLINE_DEGREE + 1))
        .map(|index| domain[0] + (index as isize - SPLINE_DEGREE as isize) as f32 * interval_width)
        .collect()
}

pub(crate) fn silu(input: f32) -> f32 {
    input / (1.0 + (-input).exp())
}

pub(crate) fn silu_derivative(input: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-input).exp());
    sigmoid * (1.0 + input * (1.0 - sigmoid))
}

fn basis_values(knots: &[f32], degree: usize, input: f32) -> Vec<f32> {
    if !input.is_finite() || input < knots[0] || input >= knots[knots.len() - 1] {
        return vec![0.0; knots.len() - degree - 1];
    }

    let mut values: Vec<f32> = knots
        .windows(2)
        .map(|window| f32::from(window[0] <= input && input < window[1]))
        .collect();

    for current_degree in 1..=degree {
        let mut next = vec![0.0; values.len() - 1];
        for index in 0..next.len() {
            let left_denominator = knots[index + current_degree] - knots[index];
            let right_denominator = knots[index + current_degree + 1] - knots[index + 1];
            let left = if left_denominator == 0.0 {
                0.0
            } else {
                (input - knots[index]) * values[index] / left_denominator
            };
            let right = if right_denominator == 0.0 {
                0.0
            } else {
                (knots[index + current_degree + 1] - input) * values[index + 1] / right_denominator
            };
            next[index] = left + right;
        }
        values = next;
    }

    values
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::{coefficient_count, exterior_uniform_knots, SplineActivation};

    #[test]
    fn exterior_uniform_knots_match_the_paper_grid_contract() {
        assert_eq!(coefficient_count(3), 6);
        let knots = exterior_uniform_knots(3, [-1.0, 1.0]);
        let expected = [
            -3.0,
            -7.0 / 3.0,
            -5.0 / 3.0,
            -1.0,
            -1.0 / 3.0,
            1.0 / 3.0,
            1.0,
            5.0 / 3.0,
            7.0 / 3.0,
            3.0,
        ];
        for (actual, expected) in knots.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn cubic_basis_is_a_partition_of_unity_on_the_base_domain() {
        let activation = SplineActivation::new(3, [-1.0, 1.0], 0.0, 1.0, vec![0.0; 6]);
        for input in [-1.0, -0.75, -0.2, 0.0, 0.4, 0.99] {
            let basis = activation.basis(input);
            let sum: f32 = basis.values.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-6, "input={input}, sum={sum}");
            assert!(basis.values.iter().all(|value| *value >= 0.0));
            assert!(basis.values.iter().filter(|value| **value > 0.0).count() <= 4);
        }
    }

    #[test]
    fn constant_coefficients_reproduce_a_constant_only_on_spline_support() {
        let activation = SplineActivation::new(3, [-1.0, 1.0], 0.0, 1.0, vec![1.0; 6]);
        for input in [-1.0, -0.2, 0.4, 1.0] {
            let (value, _) = activation.evaluate(input);
            assert!((value - 1.0).abs() < 1.0e-6);
        }
        for input in [-3.1, 3.0, 3.1] {
            let (value, basis) = activation.evaluate(input);
            assert_eq!(value, 0.0);
            assert!(basis.values.iter().all(|basis_value| *basis_value == 0.0));
            assert!(basis
                .derivatives
                .iter()
                .all(|basis_derivative| *basis_derivative == 0.0));
        }
    }

    #[test]
    fn cubic_basis_has_the_expected_left_boundary_values() {
        let activation = SplineActivation::new(3, [-1.0, 1.0], 0.0, 1.0, vec![0.0; 6]);
        let basis = activation.basis(-1.0);
        let expected = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0, 0.0, 0.0];
        for (actual, expected) in basis.values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn spline_values_and_derivatives_match_scipy_with_identical_knots() {
        // Generated with SciPy 1.18.0 BSpline using this module's explicit
        // exterior-extended knot vector, not a clamped-knot constructor.
        let activation = SplineActivation::new(
            3,
            [-1.0, 1.0],
            0.0,
            1.0,
            vec![0.25, -0.5, 0.75, 1.25, -1.0, 0.5],
        );
        let fixtures = [
            (-1.0, -0.166_666_67, 0.375),
            (-0.75, 0.043_538_41, 1.209_960_9),
            (-0.2, 0.782_333_3, 1.027_5),
            (0.17, 0.928_409_34, -0.391_912_5),
            (0.4, 0.691_5, -1.676_25),
            (1.0, -0.375, -0.562_5),
        ];

        for (input, expected_value, expected_derivative) in fixtures {
            let basis = activation.basis(input);
            let (actual_value, actual_derivative) = activation.spline_value_and_derivative(&basis);
            assert!((actual_value - expected_value).abs() < 1.0e-6);
            assert!((actual_derivative - expected_derivative).abs() < 1.0e-5);
        }
    }
}
