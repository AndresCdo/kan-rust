use crate::network::{Layer, Network};
use crate::data_structures::{Matrix, Vector};

#[test]
fn test_network_forward() {
    let layer1 = Layer::new(
        Matrix::new(vec![Vector::new(vec![0.2, 0.8]), Vector::new(vec![0.6, 0.4])]),
        Vector::new(vec![0.1, 0.2])
    );

    let layer2 = Layer::new(
        Matrix::new(vec![Vector::new(vec![0.5, 0.3]), Vector::new(vec![0.9, 0.7])]),
        Vector::new(vec![0.0, 0.1])
    );

    let network = Network::new(vec![layer1, layer2]);

    let input = Vector::new(vec![1.0, 2.0]);
    let output = network.forward(input);

    assert_eq!(output.len(), 2); // Assuming the network structure and weights produce 2 outputs
}
