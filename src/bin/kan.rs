use kan::{data_structures::*, network::Network};
use rand::{thread_rng, Rng};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use std::error::Error;
use serde_json::{to_writer, Error as JsonError};
use std::fs::{File, OpenOptions};

const LEARNING_RATE: f32 = 0.1;
const NUM_EPOCHS: u32 = 100;
const INPUT_SIZE: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    /*T his project is a Rust implementation of a Kolmogorov–Arnold Network (KAN) neural network. */
    // Initialize random weight matrix with scaled values
    let mut weights = Matrix::random(INPUT_SIZE, INPUT_SIZE);
    weights.iter_mut().for_each(|x | *x *= 0.01);

    // Create random bias vector
    let biases = Vector::random(INPUT_SIZE);

    // Build the neural network layer
    let layer = Layer::new(weights, biases);
    let mut model = Network::new(vec![layer]);

    // Setup progress bar with custom style
    let progress_bar = ProgressBar::new(NUM_EPOCHS as u64).with_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .progress_chars("#>-"),
    );

    progress_bar.set_message("Training...");


    // Generate random input and target vectors
    let input = Vector::random(INPUT_SIZE);
    let target = Vector::random(INPUT_SIZE);

    // Training loop
    for _ in 0..NUM_EPOCHS {
        // Train the model on the input and target
        model.train(input.clone(), target.clone(), LEARNING_RATE);

        // Increment the progress bar
        progress_bar.inc(1);
    }
    progress_bar.finish();
    // Evaluate the model on a test set from the input and target
    let test_input = Vector::new(input.elements.iter().map(|&x| x + 0.1).collect());
    let test_target = Vector::new(target.elements.iter().map(|&x| x + 0.1).collect());
    let output = model.predict(test_input.clone());

    // println!("Test Input: {:?}", test_input);
    // println!("Test Target: {:?}", test_target);
    // println!("Output: {:?}", output);

    // Serialize the model to a JSON file
    save_model(&model, "model.json")?;

    println!("Model saved to model.json");

    Ok(())
    }

// Function to save the model to a JSON file
fn save_model(model: &Network, path: &str) -> Result<(), JsonError> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();


    to_writer(file, model)?;
    Ok(())
}
