use kan::data_structures::{Matrix, Vector};
use kan::network::{Layer, Network};
use rand::Rng;
use indicatif::ProgressBar;
use ctrlc;
use std::result::Result;
use std::sync::{Arc, Mutex};
use std::fs::File; // Import the File type

const LEARNING_RATE: f64 = 0.1;
const NUM_TRAINING_ITERATIONS: u32 = 10000;
const INPUT_SIZE: usize = 10;

fn main() {
    // Initialize the network model, either by loading from a file or creating a new one
    let mut weights = Matrix::random(INPUT_SIZE, INPUT_SIZE);
    for row in &mut weights.rows {
        for element in &mut row.elements {
            *element = *element * 0.1;
        }
    }
    let biases = Vector::random(INPUT_SIZE);
    let layer = Layer::new(weights, biases);
    let mut model = Network::new(vec![layer]); // Declare `model` as mutable

    // Create a progress bar for tracking training progress
    let pb = ProgressBar::new(NUM_TRAINING_ITERATIONS as u64);

    // Training loop
    while true {
        pb.reset();
        // Set up Ctrl-C handler to save the model before exiting

        for _ in 0..NUM_TRAINING_ITERATIONS {
            // Generate a random input vector
            let input = Vector::random(INPUT_SIZE);

            // Compute the target output
            let targets = input.clone();

            for layer in &mut model.layers {
                // Compute the output of the layer
                let output = layer.forward(&input);

                // Compute the error and gradient
                let (weight_gradients, delta) = layer.backward(&input, &output, &targets);

                // Update the weights and biases
                layer.update(&weight_gradients, &delta, LEARNING_RATE);
            }
            
            pb.inc(1);
        }

        // Check if the user has pressed Ctrl-C
        let model_clone = model.clone();
        if ctrlc::set_handler(move || {
            // Save the model to a file
            let model_file = File::create("model.json").unwrap();
            serde_json::to_writer(model_file, &model_clone).unwrap();
            std::process::exit(0);
        }).is_err() {
            println!("Error setting Ctrl-C handler");
            break;
        }
    }

    pb.finish();

    // Save the model to a file
    let model_file = File::create("model.json").unwrap();
    serde_json::to_writer(model_file
        , &model).unwrap();

    println!("Model saved to model.json");
}