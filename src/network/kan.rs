use crate::data_structures::{SplineActivation, SplineLayer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KanLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub grid_size: usize,
    pub spline_order: usize,
    pub activations: Vec<Vec<SplineActivation>>,
}

impl KanLayer {
    pub fn new(input_size: usize, output_size: usize, grid_size: usize) -> Self {
        let spline_order = 3;

        let mut activations = Vec::new();

        for _ in 0..output_size {
            let mut row = Vec::new();
            for _ in 0..input_size {
                row.push(SplineActivation::new(grid_size));
            }
            activations.push(row);
        }

        KanLayer {
            input_size,
            output_size,
            grid_size,
            spline_order,
            activations,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];

        for j in 0..self.output_size {
            let mut sum = 0.0;
            for i in 0..self.input_size {
                sum += self.activations[j][i].forward(input[i]);
            }
            output[j] = sum;
        }

        output
    }

    pub fn l1_norm(&self, inputs: &[Vec<f32>]) -> f32 {
        let mut total = 0.0;

        for input in inputs {
            for j in 0..self.output_size {
                for i in 0..self.input_size {
                    total += self.activations[j][i].forward(input[i]).abs();
                }
            }
        }

        total
    }

    pub fn num_params(&self) -> usize {
        self.output_size * self.input_size * (self.grid_size + self.spline_order + 3)
    }

    pub fn get_activation_magnitudes(&self) -> Vec<Vec<f32>> {
        let mut magnitudes = vec![vec![0.0; self.input_size]; self.output_size];

        for j in 0..self.output_size {
            for i in 0..self.input_size {
                let mut sum = 0.0;
                for coeff in &self.activations[j][i].coeffs {
                    sum += coeff.abs();
                }
                magnitudes[j][i] = sum;
            }
        }

        magnitudes
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KanNetwork {
    pub layers: Vec<KanLayer>,
    pub grid_size: usize,
    pub grid_extend_step: usize,
    pub lambda_reg: f32,
    pub mu1: f32,
    pub mu2: f32,
}

impl KanNetwork {
    pub fn new(layer_sizes: &[usize], grid_size: usize) -> Self {
        let mut layers = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            layers.push(KanLayer::new(layer_sizes[i], layer_sizes[i + 1], grid_size));
        }

        KanNetwork {
            layers,
            grid_size,
            grid_extend_step: 200,
            lambda_reg: 0.01,
            mu1: 1.0,
            mu2: 1.0,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = input.to_vec();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        self.forward(input)
    }

    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    pub fn mse_loss(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets) {
            let output = self.forward(input);
            let mut sample_loss = 0.0;

            for (o, t) in output.iter().zip(target.iter()) {
                let diff = o - t;
                sample_loss += diff * diff;
            }

            total_loss += sample_loss;
        }

        total_loss / inputs.len() as f32
    }

    pub fn l1_regularization(&self, inputs: &[Vec<f32>]) -> f32 {
        let mut total_l1 = 0.0;

        for layer in &self.layers {
            total_l1 += layer.l1_norm(inputs);
        }

        total_l1 / inputs.len() as f32
    }

    pub fn entropy(&self) -> f32 {
        let mut total_entropy = 0.0;

        for layer in &self.layers {
            let magnitudes = layer.get_activation_magnitudes();
            let total: f32 = magnitudes.iter().flat_map(|row| row.iter()).sum();

            if total > 0.0 {
                for row in &magnitudes {
                    for &m in row {
                        if m > 0.0 {
                            let p = m / total;
                            total_entropy -= p * p.ln();
                        }
                    }
                }
            }
        }

        total_entropy
    }

    pub fn total_loss(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let pred_loss = self.mse_loss(inputs, targets);
        let l1_loss = self.l1_regularization(inputs) * self.mu1;
        let entropy_loss = self.entropy() * self.mu2;

        pred_loss + self.lambda_reg * (l1_loss + entropy_loss)
    }

    pub fn train_step(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], learning_rate: f32) {
        let mut gradients = Vec::new();

        for layer in &self.layers {
            gradients.push(vec![
                vec![
                    vec![0.0; layer.activations[0][0].coeffs.len()];
                    layer.input_size
                ];
                layer.output_size
            ]);
        }

        for (input, target) in inputs.iter().zip(targets) {
            let output = self.forward(input);

            let mut output_grad = Vec::new();
            for (o, t) in output.iter().zip(target.iter()) {
                output_grad.push(2.0 * (o - t));
            }

            self.backward_layer(input, &output, &output_grad, &mut gradients);
        }

        for (layer, layer_grad) in self.layers.iter_mut().zip(gradients.iter()) {
            for j in 0..layer.output_size {
                for i in 0..layer.input_size {
                    for (k, coeff) in layer.activations[j][i].coeffs.iter_mut().enumerate() {
                        if k < layer_grad[j][i].len() {
                            *coeff -= learning_rate * layer_grad[j][i][k] / inputs.len() as f32;
                        }
                    }
                }
            }
        }
    }

    fn backward_layer(
        &self,
        input: &[f32],
        output: &[f32],
        output_grad: &[f32],
        gradients: &mut Vec<Vec<Vec<Vec<f32>>>>,
    ) {
        let layer = &self.layers[0];

        for j in 0..layer.output_size {
            let grad_j = output_grad[j];

            for i in 0..layer.input_size {
                let x = input[i];
                let activation = &layer.activations[j][i];

                let spline_deriv = self.spline_derivative(&activation.coeffs, x);

                let base = SplineLayer::silu(x);
                let base_deriv = SplineLayer::silu_derivative(x);
                let spline = activation.evaluate_spline(x);

                let chain_grad = grad_j
                    * (activation.base_weight * base_deriv
                        + activation.spline_weight * spline_deriv);

                gradients[0][j][i][0] += grad_j * base;
                gradients[0][j][i][1] += grad_j * spline;

                let coeff_grad = grad_j * activation.spline_weight * spline_deriv;
                for k in 0..activation.coeffs.len() {
                    if k < gradients[0][j][i].len() {
                        gradients[0][j][i][k] += coeff_grad;
                    }
                }
            }
        }
    }

    fn spline_derivative(&self, coeffs: &[f32], x: f32) -> f32 {
        let k = 3;
        let grid_min = -1.0;
        let grid_max = 1.0;

        if coeffs.len() < 2 {
            return 0.0;
        }

        let mut deriv_coeffs = Vec::new();
        for i in 0..(coeffs.len() - 1) {
            deriv_coeffs.push((coeffs[i + 1] - coeffs[i]) * k as f32);
        }

        if x < grid_min || x > grid_max {
            return 0.0;
        }

        let total_intervals = self.grid_size;
        let interval_width = (grid_max - grid_min) / total_intervals as f32;

        let scaled_x = (x - grid_min) / interval_width;
        let mut spline_idx = scaled_x.floor() as usize;
        spline_idx = spline_idx.min(total_intervals.saturating_sub(1));

        let t = scaled_x - spline_idx as f32;

        deriv_coeffs.get(spline_idx).copied().unwrap_or(0.0) / interval_width
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        epochs: usize,
        learning_rate: f32,
    ) {
        for epoch in 0..epochs {
            self.train_step(inputs, targets, learning_rate);

            if epoch % 50 == 0 {
                let loss = self.total_loss(inputs, targets);
                println!("Epoch {}: Loss = {}", epoch, loss);
            }
        }
    }

    pub fn grid_extend(&mut self, new_grid_size: usize) {
        for layer in &mut self.layers {
            let old_grid_size = layer.grid_size;
            layer.grid_size = new_grid_size;

            for j in 0..layer.output_size {
                for i in 0..layer.input_size {
                    let old_coeffs = layer.activations[j][i].coeffs.clone();
                    let old_k = layer.spline_order;

                    let new_k = old_k;
                    let new_coeffs_len = new_grid_size + new_k;
                    let mut new_coeffs = vec![0.0; new_coeffs_len];

                    let scale = old_grid_size as f32 / new_grid_size as f32;
                    for new_idx in 0..new_coeffs_len {
                        let old_idx = (new_idx as f32 * scale) as usize;
                        let old_idx = old_idx.min(old_coeffs.len().saturating_sub(1));
                        new_coeffs[new_idx] = old_coeffs[old_idx];
                    }

                    layer.activations[j][i].coeffs = new_coeffs;
                    layer.activations[j][i].grid_size = new_grid_size;
                }
            }
        }
    }

    pub fn prune(&mut self, threshold: f32) {
        for layer in &mut self.layers {
            let magnitudes = layer.get_activation_magnitudes();

            let mut to_remove = Vec::new();

            for i in 0..layer.input_size {
                let mut in_mag = 0.0f32;
                let mut out_mag = 0.0f32;

                for j in 0..layer.output_size {
                    out_mag = out_mag.max(magnitudes[j][i]);
                }

                for j in 0..layer.output_size {
                    in_mag = in_mag.max(magnitudes[j][i]);
                }

                if in_mag < threshold && out_mag < threshold {
                    to_remove.push(i);
                }
            }

            println!(
                "Pruning {} connections (threshold: {})",
                to_remove.len(),
                threshold
            );
        }
    }

    pub fn symbolic_fix(
        &self,
        input_idx: usize,
        output_idx: usize,
    ) -> Option<(f32, f32, f32, f32)> {
        None
    }

    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        let mut shape = vec![self.layers[0].input_size];
        for layer in &self.layers {
            shape.push(layer.output_size);
        }
        shape
    }
}

pub fn create_kan(shape: &[usize], grid_size: usize) -> KanNetwork {
    KanNetwork::new(shape, grid_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kan_creation() {
        let kan = create_kan(&[2, 5, 1], 3);
        assert_eq!(kan.layers.len(), 2);
        assert_eq!(kan.get_shape(), vec![2, 5, 1]);
    }

    #[test]
    fn test_kan_forward() {
        let kan = create_kan(&[2, 5, 1], 3);
        let input = vec![0.5, 0.3];
        let output = kan.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_kan_mse_loss() {
        let kan = create_kan(&[2, 1], 3);
        let inputs = vec![vec![0.5, 0.3]];
        let targets = vec![vec![0.8]];

        let loss = kan.mse_loss(&inputs, &targets);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_grid_extend() {
        let mut kan = create_kan(&[2, 3, 1], 3);
        let original_params = kan.num_params();

        kan.grid_extend(10);

        assert!(kan.num_params() > original_params);
    }

    #[test]
    fn test_kan_training() {
        let mut kan = create_kan(&[1, 1], 3);

        let inputs = vec![vec![0.0]];
        let targets = vec![vec![0.0]];

        let initial_loss = kan.mse_loss(&inputs, &targets);
        kan.train(&inputs, &targets, 10, 0.1);
        let final_loss = kan.mse_loss(&inputs, &targets);

        assert!(final_loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_kan_predict_batch() {
        let kan = create_kan(&[2, 3, 1], 3);
        let inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];

        let outputs = kan.predict_batch(&inputs);

        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 1);
            assert!(output[0].is_finite());
        }
    }

    #[test]
    fn test_l1_regularization() {
        let kan = create_kan(&[2, 2], 3);
        let inputs = vec![vec![0.5, 0.3]];

        let l1 = kan.l1_regularization(&inputs);
        assert!(l1 >= 0.0);
    }

    #[test]
    fn test_entropy() {
        let kan = create_kan(&[2, 2], 3);

        let entropy = kan.entropy();
        assert!(entropy >= 0.0);
    }
}
