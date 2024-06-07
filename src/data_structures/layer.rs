
use crate::data_structures::{matrix, Matrix, Vector};
use serde::{de, Deserialize, Serialize};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// A layer in a neural network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// The weights of the layer.
    pub weights: Matrix,
    /// The biases of the layer.
    pub biases: Vector,
}

impl Layer {
    /// Creates a new layer with the given weights and biases.
    pub fn new(weights: Matrix, biases: Vector) -> Self {
        Layer { weights, biases }
    }

    /// Creates a new layer with random weights and biases.
    pub fn random(input_size: usize, output_size: usize) -> Self {
        Layer {
            weights: Matrix::random(input_size, output_size),
            biases: Vector::random(output_size),
        }
    }

    /// Performs the forward propagation of the layer.
    pub fn forward(&self, input: &Vector) -> Vector {
        // Compute the dot product of weights and input, then add biases
        // println!("Forward");
        // println!("input: {:?}", input);
        // println!("weights: {:?}", self.weights);
        // println!("biases: {:?}", self.biases);
        let output = self.weights.multiply_with_vector(input).unwrap().add(&self.biases);


        // println!("output: {:?}", output);
        
        output
    }

    /// Performs the backward propagation of the layer.
    pub fn backward(&self, input: &Vector, output: &Vector, target: &Vector) -> (Matrix, Vector) {
        println!("Backward");
        // println!("input: {:?}", input);
        // println!("target: {:?}", target);
        // Compute the error and gradient
        println!("output: {:?}", output);
        let error = output.subtract(target);
        println!("error: {:?}", error);
        let gradient = output.elementwise_multiply(&output.subtract(&Vector::ones(output.len())));
        // println!("gradient: {:?}", gradient);
        let delta = self.delta(&error, &gradient);
        // println!("delta: {:?}", delta);
        let weight_gradients = self.weight_gradients(input, output, &delta);
        // println!("weight_gradients: {:?}", weight_gradients);
        (weight_gradients, delta)
    }


    /// Updates the weights and biases of the layer based on the gradients and learning rate.
    pub fn update(&mut self, weight_gradients: &Matrix, delta: &Vector, learning_rate: f32) {
        self.weights = weight_gradients.scalar_multiply(learning_rate).add(&self.weights).unwrap();
        self.biases = delta.scalar_multiply(learning_rate).add(&self.biases);
    }

    /// Trains the layer on a single input and target.
    pub fn train(&mut self, input: &Vector, target: &Vector, learning_rate: f32) {
        let output = self.forward(input);
        let (weight_gradients, delta) = self.backward(input, &output, target);
        self.update(&weight_gradients, &delta, learning_rate);
    }

    /// Predicts the output of the layer for a given input.
    pub fn predict(&self, input: &Vector) -> Vector {
        self.forward(input)
    }

    /// Calculates the loss between the output and target.
    pub fn loss(&self, input: &Vector, target: &Vector) -> f32 {
        let output = self.forward(input);
        let error = output.subtract(target);
        error.elementwise_multiply(&error).sum()
    }

    /// Calculates the accuracy of the output compared to the target.
    pub fn accuracy(&self, input: &Vector, target: &Vector) -> f32 {
        let output = self.forward(input);
        let error = output.subtract(target);
        let correct = error.elements.iter().filter(|&&x| x.abs() < 0.5).count() as f32;
        correct / target.len() as f32
    }

    /// Evaluates the layer on a batch of inputs and targets.
    pub fn evaluate(&self, inputs: &[Vector], targets: &[Vector]) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        for (input, target) in inputs.iter().zip(targets) {
            total_loss += self.loss(input, target);
            total_accuracy += self.accuracy(input, target);
        }
        (
            total_loss / inputs.len() as f32,
            total_accuracy / inputs.len() as f32,
        )
    }

    /// Trains the layer on a batch of inputs and targets for one epoch.
    pub fn train_epoch(&mut self, inputs: &[Vector], targets: &[Vector], learning_rate: f32) {
        for (input, target) in inputs.iter().zip(targets) {
            self.train(input, target, learning_rate);
        }
    }

    /// Trains the layer on a batch of inputs and targets for multiple epochs.
    pub fn train_epochs(
        &mut self,
        inputs: &[Vector],
        targets: &[Vector],
        learning_rate: f32,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            self.train_epoch(inputs, targets, learning_rate);
        }
    }

    /// Trains the layer until convergence or a maximum number of epochs is reached.
    pub fn train_until_convergence(
        &mut self,
        inputs: &[Vector],
        targets: &[Vector],
        learning_rate: f32,
        max_epochs: usize,
        tolerance: f32,
    ) {
        let mut epoch = 0;
        let mut prev_loss = f32::INFINITY;
        let mut loss = self.evaluate(inputs, targets).0;
        while (prev_loss - loss).abs() > tolerance && epoch < max_epochs {
            prev_loss = loss;
            self.train_epoch(inputs, targets, learning_rate);
            loss = self.evaluate(inputs, targets).0;
            epoch += 1;
        }
    }

    /// Predicts the output for a batch of inputs.
    pub fn predict_batch(&self, inputs: &[Vector]) -> Vec<Vector> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    /// Calculates the average loss for a batch of inputs and targets.
    pub fn loss_batch(&self, inputs: &[Vector], targets: &[Vector]) -> f32 {
        let total_loss: f32 =
            inputs.iter().zip(targets).map(|(input, target)| self.loss(input, target)).sum();
        total_loss / inputs.len() as f32
    }

    /// Calculates the average accuracy for a batch of inputs and targets.
    pub fn accuracy_batch(&self, inputs: &[Vector], targets: &[Vector]) -> f32 {
        let total_accuracy: f32 = inputs
            .iter()
            .zip(targets)
            .map(|(input, target)| self.accuracy(input, target))
            .sum();
        total_accuracy / inputs.len() as f32
    }

    /// Evaluates the layer on a batch of inputs and targets.
    pub fn evaluate_batch(&self, inputs: &[Vector], targets: &[Vector]) -> (f32, f32) {
        (self.loss_batch(inputs, targets), self.accuracy_batch(inputs, targets))
    }

    pub fn from_str(s: &str) -> Self {
        let mut lines = s.lines();
        let weights = Matrix::from_string(lines.next().unwrap()).unwrap();  
        let biases = Vector::from_string(lines.next().unwrap())
            .unwrap();  
        Layer { weights, biases }
    }

    pub fn to_str(&self) -> String {
        let weights = self.weights.to_string();
        let biases = self.biases.to_string();
        format!("{}\n{}\n", weights, biases)
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        self.weights = self.weights.scalar_multiply(learning_rate);
    }

    pub fn update_biases(&mut self, learning_rate: f32) {
        self.biases = self.biases.scalar_multiply(learning_rate);
    }

    pub fn delta(&self, error: &Vector, gradient: &Vector) -> Vector {
        error.elementwise_multiply(gradient)
    }

    pub fn weight_gradients(&self, input: &Vector, output: &Vector, gradient: &Vector) -> Matrix {
        let cols = self.weights.col_count();
        let rows = self.weights.row_count();
        let mut weight_gradients = Matrix::zeros(cols, rows);
        for i in 0..cols {
            for j in 0..rows {
                weight_gradients.set_element(i, j, input.get_element(i) * gradient.get_element(j)).unwrap();
            }
        }
        weight_gradients
    }
}
