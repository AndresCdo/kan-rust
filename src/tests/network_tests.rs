use crate::network::Network;
use crate::data_structures::{Matrix, Vector, Layer};


#[test]
fn test_network_forward() {
    let layer1 = Layer::new(
        Matrix::new(vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![3.0, 4.0]),
        ]),
        Vector::new(vec![1.0, 2.0]),
    );
    let layer2 = Layer::new(
        Matrix::new(vec![
            Vector::new(vec![5.0, 6.0]),
            Vector::new(vec![7.0, 8.0]),
        ]),
        Vector::new(vec![3.0, 4.0]),
    );
    let network = Network::new(vec![layer1, layer2]);
    let input = Vector::new(vec![1.0, 2.0]);
    let output = network.forward(input);
    // Test that it runs without panicking
    assert!(true);

    // Test that the output has the correct shape
    assert_eq!(output.len(), 2);
}

#[test]
fn test_network_backward() {
    let layer1 = Layer::new(
        Matrix::new(vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![3.0, 4.0]),
        ]),
        Vector::new(vec![1.0, 2.0]),
    );
    let layer2 = Layer::new(
        Matrix::new(vec![
            Vector::new(vec![5.0, 6.0]),
            Vector::new(vec![7.0, 8.0]),
        ]),
        Vector::new(vec![3.0, 4.0]),
    );
    let network = Network::new(vec![layer1, layer2]);
    let input = Vector::new(vec![1.0, 2.0]);
    let output = network.forward(input);
    let target = Vector::new(vec![1.0, 2.0]);
    let error = network.backward(output, target);
    
    // Test that it runs without panicking
    assert!(true);
    
    // Test that the error has the correct shape
    assert_eq!(error.1.len(), 2);
}