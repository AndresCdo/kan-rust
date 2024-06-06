# Kolmogorov–Arnold Network (KAN) in Rust

This project is a Rust implementation of a Kolmogorov–Arnold Network (KAN) neural network. The KAN network is a type of feedforward neural network that uses a spline activation function to approximate any continuous function. The network is trained using backpropagation and gradient descent to minimize the loss function. The project includes a library for building and training the network, as well as an example application that demonstrates how to use the network to solve a regression problem.

## Description

- `src/bin/kan.rs`: The main entry point of the application.
- `src/data_structures`: Contains various data structures used in the project like `KANLayer`, `layer`, `matrix`, `spline`, and `vector`.
- `src/lib.rs`: The library file.
- `src/network`: Contains the network implementation.
- `src/tests`: Contains the unit tests for the various components of the project.
- `src/utils`: Contains utility functions and modules like `activations`, `is_close_enough`, and `loss_functions`.
- `model` and `model.json`: These files are related to the model used in the project.

## How to Run

To run the project, use the following command:

```bash
cargo run
```

## How to Test

To run the tests, use the following command:

```bash
cargo test
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Author

[Andres Caicedo](https://github.com/AndresCdo)

## Acknowledgements

- [Rust Programming Language](https://www.rust-lang.org/)
- [Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v1)

## Future Work

- Implement more advanced features like dropout and batch normalization.
- Optimize the code for better performance.
- Explore different applications of the KAN network.
- Add more unit tests and integration tests.
- Create a more user-friendly interface for training and using the network.
- Implement a GUI for visualizing the network and its results.

## Contributions

Contributions are welcome! Please feel free to submit pull requests or open issues.
- **Add more activation functions:** Implement more activation functions like ReLU, sigmoid, and tanh.
- **Implement different loss functions:** Implement different loss functions like mean squared error, cross-entropy, and hinge loss.
- **Add support for different datasets:** Add support for different datasets like MNIST, CIFAR-10, and ImageNet.
- **Implement different optimization algorithms:** Implement different optimization algorithms like Adam, SGD, and RMSprop.
- **Improve the documentation:** Improve the documentation of the code and the project.
- **Add more examples:** Add more examples of how to use the KAN network.
- **Create a GUI:** Create a GUI for visualizing the network and its results.

This project is still under development, but I hope it will be a useful resource for anyone interested in learning about KAN neural networks and implementing them in Rust.
