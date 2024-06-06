
use crate::data_structures::{Layer, Matrix, Vector};

#[test]
fn test_new_layer() {
    let weights = Matrix::ones(2, 3);
    let biases = Vector::zeros(3);
    let layer = Layer::new(weights.clone(), biases.clone());
    assert_eq!(layer.weights, weights);
    assert_eq!(layer.biases, biases);
}

#[test]
fn test_random_layer() {
    let input_size = 2;
    let output_size = 3;
    let layer = Layer::random(input_size, output_size);
    assert_eq!(layer.weights.row_count(), input_size);
    assert_eq!(layer.weights.col_count(), output_size);
    assert_eq!(layer.biases.len(), output_size);
}

// #[test]
// fn test_forward() {
//     let weights = Matrix::from_vec(vec![vec![1.0, 2.0]]);
//     let biases = Vector::from_vec(vec![1.0, 2.0]);
//     let layer = Layer::new(weights, biases);
//     let input = Vector::from_vec(vec![1.0, 2.0]);
//     let output = layer.forward(&input);
//     assert_eq!(output, Vector::from_vec(vec![9.0, 12.0]));
// }

// #[test]
// fn test_backward() {
//     let weights = Matrix::from_vec(vec![vec![1.0, 2.0]]);
//     let biases = Vector::from_vec(vec![1.0, 2.0]);
//     let layer = Layer::new(weights, biases);
//     let input = Vector::from_vec(vec![1.0, 2.0]);
//     let output = Vector::from_vec(vec![9.0, 12.0]);
//     let target = Vector::from_vec(vec![1.0, 0.0]);
//     let (weight_gradients, bias_gradients) = layer.backward(&input, &output, &target);
//     assert_eq!(weight_gradients,  Matrix::from_vec(vec![vec![-8.0, -16.0, -12.0, -24.0]]));
//     assert_eq!(bias_gradients, Vector::from_vec(vec![-8.0, -12.0]));
// }

// #[test]
// fn test_update() {
//     let mut layer = Layer::new(Matrix::ones(2, 2), Vector::ones(2));
//     let weight_gradients = Matrix::from_vec(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
//     let delta = Vector::from_vec(vec![1.0, 2.0]);
//     let learning_rate = 0.1;
//     layer.update(&weight_gradients, &delta, learning_rate);
//     assert_eq!(layer.weights, Matrix::from_vec(vec![vec![0.9, 1.8], vec![2.7, 3.6]]));
//     assert_eq!(layer.biases, Vector::from_vec(vec![0.8, 1.8]));
// }

// #[test]
// fn test_train() {
//     let mut layer = Layer::new(Matrix::ones(2, 2), Vector::ones(2));
//     let input = Vector::from_vec(vec![1.0, 2.0]);
//     let target = Vector::from_vec(vec![1.0, 0.0]);
//     let learning_rate = 0.1;
//     layer.train(&input, &target, learning_rate);
//     assert_eq!(layer.weights, Matrix::from_vec(vec![vec![0.9, 1.8], vec![2.7, 3.6]]));
//     assert_eq!(layer.biases, Vector::from_vec(vec![0.8, 1.8]));
// }
