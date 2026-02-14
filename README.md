# Kolmogorov–Arnold Network (KAN) in Rust

This project is a Rust implementation of a **Kolmogorov–Arnold Network (KAN)** neural network, based on the paper ["KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/html/2404.19756v5) by Ziming Liu et al.

## Description

KAN is a promising alternative to Multi-Layer Perceptrons (MLPs). Unlike MLPs that have fixed activation functions on nodes, KANs have **learnable activation functions on edges** (weights), parameterized as B-splines.

### Key Features

- **Learnable activation functions** on edges instead of fixed activations on nodes
- **B-spline parametrization** for univariate functions (order k=3 by default)
- **No linear weight matrices** - every parameter is a learnable 1D function
- **Grid extension** for incremental accuracy improvement
- **Regularization** (L1 + entropy) for network simplification
- **Pruning** for interpretability
- **Symbolic fixing** for formula discovery

### Architecture

```
Input Layer (n0)
    │
    ▼
┌─────────────────────┐
│  KAN Layer 0: Φ₀    │
│  [n₀ × n₁] funciones│
│  univariadas        │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  KAN Layer 1: Φ₁    │
└─────────────────────┘
    │ ... │
    ▼
┌─────────────────────┐
│  KAN Layer L-1      │
└─────────────────────┘
    │
    ▼
Output Layer (n_L)

Cada función: ϕ(x) = w_b·SiLU(x) + w_s·spline(x)
```

## Project Structure

```
kan-rust/
├── src/
│   ├── bin/kan.rs           # Binary entry point
│   ├── data_structures/     # Core data structures
│   │   ├── vector.rs        # Vector operations
│   │   ├── matrix.rs        # Matrix operations
│   │   ├── layer.rs         # MLP layer (legacy)
│   │   └── spline.rs        # B-spline implementation
│   ├── network/             # Network implementations
│   │   ├── network.rs       # MLP network (legacy)
│   │   └── kan.rs           # KAN implementation ✓
│   ├── utils/               # Utility functions
│   └── tests/               # Unit tests
├── Cargo.toml
└── README.md
```

## Installation

```bash
cargo build
```

## Usage

### Basic Example

```rust
use kan::network::kan::create_kan;

fn main() {
    // Create a KAN with shape [2, 5, 1]
    let mut kan = create_kan(&[2, 5, 1], 3); // 3 = grid_size
    
    // Training data
    let inputs = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ];
    let targets = vec![
        vec![0.3],
        vec![0.7],
        vec![1.1],
    ];
    
    // Train
    kan.train(&inputs, &targets, 100, 0.01);
    
    // Predict
    let input = vec![0.1, 0.2];
    let output = kan.forward(&input);
    println!("Output: {:?}", output);
}
```

### Advanced: Grid Extension

```rust
// Extend grid from G=3 to G=10 for more accuracy
kan.grid_extend(10);
```

### Advanced: Pruning

```rust
// Remove connections with magnitude < 0.01
kan.prune(0.01);
```

## Mathematical Foundation

Based on the **Kolmogorov-Arnold Representation Theorem**:

$$f(x) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Each function $\phi_{q,p}$ is parameterized as a B-spline:

$$\phi(x) = w_b \cdot \text{SiLU}(x) + w_s \cdot \sum_i c_i B_i(x)$$

### Scaling Law

With B-splines of order k=3 (cubic), the approximation error scales as:

$$\text{RMSE} \propto G^{-(k+1)} = G^{-4}$$

## How to Run

```bash
cargo run
```

## How to Test

```bash
cargo test
```

## Implemented Features

| Feature | Status |
|---------|--------|
| B-spline activation functions | ✅ |
| SiLU base function | ✅ |
| Forward propagation | ✅ |
| Backpropagation | ✅ |
| Grid extension | ✅ |
| L1 regularization | ✅ |
| Entropy regularization | ✅ |
| Node pruning | ✅ |
| Symbolic fixing | 🔄 (placeholder) |

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v5) - Original paper
- [pykan](https://github.com/KindXiaoming/pykan) - Python implementation

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Author

[Andres Caicedo](https://github.com/AndresCdo)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
