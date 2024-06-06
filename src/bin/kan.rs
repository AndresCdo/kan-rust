use kan::{data_structures::*, network::Network};
use rand::{thread_rng, Rng};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use std::error::Error;
use serde_json::{to_writer, Error as JsonError};
use std::fs::{File, OpenOptions};

const LEARNING_RATE: f32 = 0.1;
const NUM_EPOCHS: u32 = 1000;
const INPUT_SIZE: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize random weight matrix with scaled values
    let mut weights = Matrix::random(INPUT_SIZE, INPUT_SIZE);
    weights.iter_mut().for_each(|x| *x *= 0.1);

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



    // Training loop
    for epoch in 0..NUM_EPOCHS {
        progress_bar.set_position(epoch as u64);
        
        // Generate random input and target vectors
        let input = Vector::random(INPUT_SIZE);
        let target = Vector::random(INPUT_SIZE);

        // Train the model on the input and target
        model.train(input, target, LEARNING_RATE);
    }

    // Evaluate the model on a test set
    let test_input = Vector::random(INPUT_SIZE);
    let test_target = Vector::random(INPUT_SIZE);
    let output = model.predict(test_input.clone());

    println!("Test Input: {:?}", test_input);
    println!("Test Target: {:?}", test_target);
    println!("Output: {:?}", output);

    // Calculate and print the loss and accuracy
    let loss = model.loss(test_input.clone(), test_target.clone());
    let accuracy = model.accuracy(test_input.clone(), test_target.clone());

    println!("Loss: {}", loss);
    println!("Accuracy: {}", accuracy);

    // // Print the weights and biases of the layer
    // println!("Weights: {:?}", model.layers[0].weights);
    // println!("Biases: {:?}", model.layers[0].biases);

    // // Print the model's structure
    // println!("Model Structure: {:?}", model);

    // // Print the model's weights and biases
    // println!("Model Weights: {:?}", model.weights());
    // println!("Model Biases: {:?}", model.biases());

    // // Print the model's layers
    // println!("Model Layers: {:?}", model.layers);

    // Print the model's output for a given input
    println!("Model Output: {:?}", model.predict(test_input.clone()));

    // Print the model's loss for a given input and target
    println!("Model Loss: {}", model.loss(test_input.clone(), test_target.clone()));

    // Print the model's accuracy for a given input and target
    println!("Model Accuracy: {}", model.accuracy(test_input.clone(), test_target.clone()));

    // Print the model's evaluation metrics for a given set of inputs and targets
    let inputs = vec![Vector::random(INPUT_SIZE), Vector::random(INPUT_SIZE)];
    let targets = vec![Vector::random(INPUT_SIZE), Vector::random(INPUT_SIZE)];
    let (loss, accuracy) = model.evaluate(&inputs, &targets);
    println!("Model Loss: {}", loss);
    println!("Model Accuracy: {}", accuracy);


    // Serialize the model to a JSON file
    save_model(&model, "model.json")?;

    println!("Model saved to model.json");

    Ok(())
    }

//     progress_bar.finish_with_message("Training complete!");

//     // Serialize the model to a JSON file
//     save_model(&model, "model.json")?;

//     println!("Model saved to model.json");

//     Ok(())
// }

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
