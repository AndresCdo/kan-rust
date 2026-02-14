use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub grid_size: usize,
    pub spline_order: usize,
    pub base_weight: Vec<Vec<f32>>,
    pub spline_weight: Vec<Vec<f32>>,
    pub spline_coeffs: Vec<Vec<Vec<f32>>>,
    pub grid_min: f32,
    pub grid_max: f32,
}

impl SplineLayer {
    pub fn new(input_size: usize, output_size: usize, grid_size: usize) -> Self {
        let spline_order = 3;
        let grid_min = -1.0;
        let grid_max = 1.0;

        let mut rng = rand::thread_rng();

        let mut base_weight = Vec::new();
        let mut spline_weight = Vec::new();
        let mut spline_coeffs = Vec::new();

        for _ in 0..output_size {
            let mut row_base = Vec::new();
            let mut row_spline = Vec::new();
            let mut row_coeffs = Vec::new();

            for _ in 0..input_size {
                row_base.push(0.0);
                row_spline.push(1.0);

                let mut coeffs = Vec::new();
                for _ in 0..(grid_size + spline_order) {
                    coeffs.push(rng.gen_range(-0.1..0.1));
                }
                row_coeffs.push(coeffs);
            }

            base_weight.push(row_base);
            spline_weight.push(row_spline);
            spline_coeffs.push(row_coeffs);
        }

        SplineLayer {
            input_size,
            output_size,
            grid_size,
            spline_order,
            base_weight,
            spline_weight,
            spline_coeffs,
            grid_min,
            grid_max,
        }
    }

    pub fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    pub fn silu_derivative(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        sig * (1.0 + x * (1.0 - sig))
    }

    pub fn evaluate_spline(&self, coeffs: &[f32], x: f32) -> f32 {
        if x < self.grid_min || x > self.grid_max {
            let scale = if x < self.grid_min {
                coeffs.first().copied().unwrap_or(0.0)
            } else {
                coeffs.last().copied().unwrap_or(0.0)
            };
            return scale;
        }

        let total_intervals = self.grid_size;
        let interval_width = (self.grid_max - self.grid_min) / total_intervals as f32;

        let scaled_x = (x - self.grid_min) / interval_width;
        let mut spline_idx = scaled_x.floor() as usize;
        spline_idx = spline_idx.min(total_intervals - 1);

        let t = scaled_x - spline_idx as f32;

        let coeffs_for_interval = &coeffs[spline_idx..spline_idx + self.spline_order + 1];

        if coeffs_for_interval.len() < self.spline_order + 1 {
            return coeffs.last().copied().unwrap_or(0.0);
        }

        self.evaluate_bspline(coeffs_for_interval, t)
    }

    fn evaluate_bspline(&self, coeffs: &[f32], t: f32) -> f32 {
        let k = self.spline_order;
        let n = coeffs.len() - 1;

        if k == 0 {
            return coeffs[n.min(coeffs.len() - 1)];
        }

        let mut temp = coeffs.to_vec();

        for p in 1..=k {
            for i in 0..(n - p + 1) {
                let alpha = t / (p as f32);
                temp[i] = (1.0 - alpha) * temp[i] + alpha * temp[i + 1];
            }
        }

        temp[0]
    }

    pub fn spline_derivative(&self, coeffs: &[f32], x: f32) -> f32 {
        let k = self.spline_order;

        if k == 0 {
            return 0.0;
        }

        let mut deriv_coeffs = Vec::new();
        for i in 0..(coeffs.len() - 1) {
            deriv_coeffs.push((coeffs[i + 1] - coeffs[i]) * k as f32);
        }

        let total_intervals = self.grid_size;
        let interval_width = (self.grid_max - self.grid_min) / total_intervals as f32;

        let scaled_x = (x - self.grid_min) / interval_width;
        let mut spline_idx = scaled_x.floor() as usize;
        spline_idx = spline_idx.min(total_intervals - 1);

        let t = scaled_x - spline_idx as f32;

        if deriv_coeffs.is_empty() {
            return 0.0;
        }

        let coeffs_for_interval = &deriv_coeffs[spline_idx
            ..spline_idx
                .saturating_sub(1)
                .max(0)
                .min(deriv_coeffs.len().saturating_sub(1))];

        self.evaluate_bspline(coeffs_for_interval, t) / interval_width
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];

        for j in 0..self.output_size {
            let mut sum = 0.0;

            for i in 0..self.input_size {
                let x = input[i];
                let base = Self::silu(x);
                let spline = self.evaluate_spline(&self.spline_coeffs[j][i], x);

                sum += self.base_weight[j][i] * base + self.spline_weight[j][i] * spline;
            }

            output[j] = sum;
        }

        output
    }

    pub fn backward(
        &self,
        input: &[f32],
        output_grad: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let mut base_weight_grad = vec![vec![0.0; self.input_size]; self.output_size];
        let mut spline_weight_grad = vec![vec![0.0; self.input_size]; self.output_size];
        let mut spline_coeff_grad =
            vec![
                vec![vec![0.0; self.grid_size + self.spline_order]; self.input_size];
                self.output_size
            ];

        let mut input_grad = vec![0.0; self.input_size];

        for j in 0..self.output_size {
            let grad_j = output_grad[j];

            for i in 0..self.input_size {
                let x = input[i];
                let base = Self::silu(x);
                let base_deriv = Self::silu_derivative(x);
                let spline = self.evaluate_spline(&self.spline_coeffs[j][i], x);
                let spline_deriv = self.spline_derivative(&self.spline_coeffs[j][i], x);

                base_weight_grad[j][i] = grad_j * base;
                spline_weight_grad[j][i] = grad_j * spline;

                let coeff_grad = grad_j * self.spline_weight[j][i] * spline_deriv;
                for k in 0..self.spline_coeffs[j][i].len() {
                    spline_coeff_grad[j][i][k] = coeff_grad;
                }

                input_grad[i] += grad_j
                    * (self.base_weight[j][i] * base_deriv
                        + self.spline_weight[j][i] * spline_deriv);
            }
        }

        (base_weight_grad, spline_weight_grad, spline_coeff_grad)
    }

    pub fn update_params(
        &mut self,
        base_grad: &[Vec<f32>],
        spline_weight_grad: &[Vec<f32>],
        coeff_grad: &[Vec<Vec<f32>>],
        learning_rate: f32,
    ) {
        for j in 0..self.output_size {
            for i in 0..self.input_size {
                self.base_weight[j][i] -= learning_rate * base_grad[j][i];
                self.spline_weight[j][i] -= learning_rate * spline_weight_grad[j][i];

                for k in 0..self.spline_coeffs[j][i].len() {
                    self.spline_coeffs[j][i][k] -= learning_rate * coeff_grad[j][i][k];
                }
            }
        }
    }

    pub fn num_params(&self) -> usize {
        self.output_size * self.input_size * (3 + self.grid_size + self.spline_order)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineActivation {
    pub coeffs: Vec<f32>,
    pub base_weight: f32,
    pub spline_weight: f32,
    pub grid_size: usize,
    pub spline_order: usize,
    pub grid_min: f32,
    pub grid_max: f32,
}

impl SplineActivation {
    pub fn new(grid_size: usize) -> Self {
        let spline_order = 3;
        let grid_min = -1.0;
        let grid_max = 1.0;

        let mut rng = rand::thread_rng();
        let mut coeffs = Vec::new();

        for _ in 0..(grid_size + spline_order) {
            coeffs.push(rng.gen_range(-0.1..0.1));
        }

        SplineActivation {
            coeffs,
            base_weight: 0.0,
            spline_weight: 1.0,
            grid_size,
            spline_order,
            grid_min,
            grid_max,
        }
    }

    pub fn forward(&self, x: f32) -> f32 {
        let base = SplineLayer::silu(x);
        let spline = self.evaluate_spline(x);
        self.base_weight * base + self.spline_weight * spline
    }

    pub fn evaluate_spline(&self, x: f32) -> f32 {
        if x < self.grid_min || x > self.grid_max {
            return if x < self.grid_min {
                self.coeffs.first().copied().unwrap_or(0.0)
            } else {
                self.coeffs.last().copied().unwrap_or(0.0)
            };
        }

        let total_intervals = self.grid_size;
        let interval_width = (self.grid_max - self.grid_min) / total_intervals as f32;

        let scaled_x = (x - self.grid_min) / interval_width;
        let mut spline_idx = scaled_x.floor() as usize;
        spline_idx = spline_idx.min(total_intervals.saturating_sub(1));

        let t = scaled_x - spline_idx as f32;

        let start = spline_idx;
        let end = (spline_idx + self.spline_order + 1).min(self.coeffs.len());

        if start >= end {
            return self.coeffs.last().copied().unwrap_or(0.0);
        }

        let coeffs_for_interval = &self.coeffs[start..end];

        self.evaluate_bspline(coeffs_for_interval, t)
    }

    fn evaluate_bspline(&self, coeffs: &[f32], t: f32) -> f32 {
        let k = self.spline_order;

        if k == 0 || coeffs.is_empty() {
            return coeffs.first().copied().unwrap_or(0.0);
        }

        let n = coeffs.len() - 1;
        let mut temp = coeffs.to_vec();

        for p in 1..=k {
            for i in 0..(n - p + 1).min(temp.len().saturating_sub(1)) {
                let alpha = t / p as f32;
                temp[i] = (1.0 - alpha) * temp[i] + alpha * temp[i + 1];
            }
        }

        temp[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spline_layer_creation() {
        let layer = SplineLayer::new(2, 3, 5);
        assert_eq!(layer.input_size, 2);
        assert_eq!(layer.output_size, 3);
        assert_eq!(layer.grid_size, 5);
    }

    #[test]
    fn test_spline_forward() {
        let layer = SplineLayer::new(2, 1, 3);
        let input = vec![0.0, 0.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_spline_activation() {
        let act = SplineActivation::new(5);
        let result = act.forward(0.5);
        assert!(result.is_finite());
    }
}
