use crate::data_structures::{matrix, Matrix, Vector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Vector,
}

impl Layer {
    pub fn new(weights: Matrix, biases: Vector) -> Self {
        Layer { weights, biases }
    }

    pub fn random(input_size: usize, output_size: usize) -> Self {
        Layer {
            weights: Matrix::random(input_size, output_size),
            biases: Vector::random(output_size),
        }
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        let matrix_output = self.weights.multiply(&Matrix::new(vec![input.clone()]));
        matrix_output.add(&Matrix::new(vec![self.biases.clone()])).rows[0].clone()
    }

    pub fn backward(&self, input: &Vector, output: &Vector, target: &Vector) -> (Matrix, Vector) {
        let error = output.subtract(target);
        let gradient = error.elementwise_multiply(output);
        let delta = self.weights.transpose().multiply(&Matrix::new(vec![gradient.clone()]));
        let weight_gradients = Matrix::new(vec![input.clone()]).multiply(&Matrix::new(vec![gradient]));
        (weight_gradients, delta.rows[0].clone())
    }

    pub fn update(&mut self, weight_gradients: &Matrix, delta: &Vector, learning_rate: f64) {
        self.weights = self.weights.subtract(&weight_gradients.scalar_multiply(learning_rate));
        self.biases = self.biases.subtract(&delta.scalar_multiply(learning_rate));
    }

    pub fn train(&mut self, input: &Vector, target: &Vector, learning_rate: f64) {
        let output = self.forward(input);
        let (weight_gradients, delta) = self.backward(input, &output, target);
        self.update(&weight_gradients, &delta, learning_rate);
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        self.forward(input)
    }

    pub fn loss(&self, input: &Vector, target: &Vector) -> f64 {
        let output = self.forward(input);
        let error = output.subtract(target);
        error.elementwise_multiply(&error).sum()
    }

    pub fn accuracy(&self, input: &Vector, target: &Vector) -> f64 {
        let output = self.forward(input);
        let error = output.subtract(target);
        let correct = error.elements.iter().filter(|&&x| x.abs() < 0.5).count() as f64;
        correct / target.len() as f64
    }

    pub fn evaluate(&self, inputs: &[Vector], targets: &[Vector]) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        for (input, target) in inputs.iter().zip(targets) {
            total_loss += self.loss(input, target);
            total_accuracy += self.accuracy(input, target);
        }
        (total_loss / inputs.len() as f64, total_accuracy / inputs.len() as f64)
    }

    pub fn train_epoch(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64) {
        for (input, target) in inputs.iter().zip(targets) {
            self.train(input, target, learning_rate);
        }
    }

    pub fn train_epochs(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            self.train_epoch(inputs, targets, learning_rate);
        }
    }

    pub fn train_until_convergence(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, max_epochs: usize, tolerance: f64) {
        let mut epoch = 0;
        let mut prev_loss = f64::INFINITY;
        let mut loss = self.evaluate(inputs, targets).0;
        while (prev_loss - loss).abs() > tolerance && epoch < max_epochs {
            prev_loss = loss;
            self.train_epoch(inputs, targets, learning_rate);
            loss = self.evaluate(inputs, targets).0;
            epoch += 1;
        }
    }

    pub fn predict_batch(&self, inputs: &[Vector]) -> Vec<Vector> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    pub fn loss_batch(&self, inputs: &[Vector], targets: &[Vector]) -> f64 {
        let total_loss: f64 = inputs.iter().zip(targets).map(|(input, target)| self.loss(input, target)).sum();
        total_loss / inputs.len() as f64
    }

    pub fn accuracy_batch(&self, inputs: &[Vector], targets: &[Vector]) -> f64 {
        let total_accuracy: f64 = inputs.iter().zip(targets).map(|(input, target)| self.accuracy(input, target)).sum();
        total_accuracy / inputs.len() as f64
    }

    pub fn evaluate_batch(&self, inputs: &[Vector], targets: &[Vector]) -> (f64, f64) {
        (self.loss_batch(inputs, targets), self.accuracy_batch(inputs, targets))
    }

    pub fn train_minibatch(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, batch_size: usize) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..inputs.len()).collect();
        indices.shuffle(&mut rng);

        for i in (0..inputs.len()).step_by(batch_size) {
            let batch_indices = &indices[i..(i + batch_size).min(inputs.len())];
            let batch_inputs: Vec<&Vector> = batch_indices.iter().map(|&i| &inputs[i]).collect();
            let batch_targets: Vec<&Vector> = batch_indices.iter().map(|&i| &targets[i]).collect();
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                self.train(input, target, learning_rate);
            }
        }
    }

    pub fn train_minibatches(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, batch_size: usize, epochs: usize) {
        for _ in 0..epochs {
            self.train_minibatch(inputs, targets, learning_rate, batch_size);
        }
    }

    pub fn train_until_convergence_minibatch(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, batch_size: usize, max_epochs: usize, tolerance: f64) {
        let mut epoch = 0;
        let mut prev_loss = f64::INFINITY;
        let mut loss = self.evaluate_batch(inputs, targets).0;
        while (prev_loss - loss).abs() > tolerance && epoch < max_epochs {
            prev_loss = loss;
            self.train_minibatch(inputs, targets, learning_rate, batch_size);
            loss = self.evaluate_batch(inputs, targets).0;
            epoch += 1;
        }
    }

    fn calculate_adagrad_update(&self, gradients: &Matrix, sum_squared_gradients: &mut Matrix, epsilon: f64) -> Matrix {
        *sum_squared_gradients = sum_squared_gradients.add(&gradients.elementwise_multiply(gradients));
        gradients.elementwise_divide(&sum_squared_gradients.elementwise_sqrt().add_scalar(epsilon))
    }

    pub fn train_minibatch_with_adagrad(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f64, batch_size: usize, epsilon: f64, sum_squared_weight_gradients: &mut Matrix, sum_squared_biases: &mut Matrix) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..inputs.len()).collect();
        indices.shuffle(&mut rng);

        for i in (0..inputs.len()).step_by(batch_size) {
            let batch_indices = &indices[i..(i + batch_size).min(inputs.len())];
            let batch_inputs: Vec<&Vector> = batch_indices.iter().map(|&i| &inputs[i]).collect();
            let batch_targets: Vec<&Vector> = batch_indices.iter().map(|&i| &targets[i]).collect();
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                let (weight_gradients, delta) = self.backward(input, &self.forward(input), target);
                let adagrad_weight_gradients = self.calculate_adagrad_update(&weight_gradients, sum_squared_weight_gradients, epsilon);
                let adagrad_biases = self.calculate_adagrad_update(&Matrix::new(vec![delta]), sum_squared_biases, epsilon);
                self.update(&adagrad_weight_gradients, &adagrad_biases.rows[0], learning_rate);
            }
        }
    }

    pub fn weight_gradients(&self, inputs: &[Vector], targets: &[Vector]) -> Matrix {
        let mut total_weight_gradients = Matrix::new(vec![Vector::new(vec![0.0; self.weights.col_count()]); self.weights.row_count()]);
        for (input, target) in inputs.iter().zip(targets) {
            let (weight_gradients, _) = self.backward(input, &self.forward(input), target);
            total_weight_gradients = total_weight_gradients.add(&weight_gradients);
        }
        total_weight_gradients.scalar_multiply(1.0 / inputs.len() as f64)
    }

    pub fn from_str(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn save(&self, path: &str) {
        std::fs::write(path, self.to_string()).unwrap();
    }

    pub fn load(path: &str) -> Self {
        Layer::from_str(&std::fs::read_to_string(path).unwrap())
    }
}
