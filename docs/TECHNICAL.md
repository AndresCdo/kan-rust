# KAN Rust Implementation

Technical documentation for the Kolmogorov-Arnold Networks implementation in Rust.

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Implementation Details](#implementation-details)

---

## Mathematical Foundation

### Kolmogorov-Arnold Representation Theorem

Any multivariate continuous function $f: [0,1]^n \to \mathbb{R}$ can be written as:

$$f(x) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Where $\phi_{q,p}: [0,1] \to \mathbb{R}$ and $\Phi_q: \mathbb{R} \to \mathbb{R}$ are univariate functions.

### Activation Function Parametrization

Each learnable activation function is parameterized as:

$$\phi(x) = w_b \cdot b(x) + w_s \cdot \text{spline}(x)$$

Where:
- $b(x) = \text{SiLU}(x) = \frac{x}{1 + e^{-x}}$ (base function)
- $\text{spline}(x) = \sum_i c_i B_i(x)$ (B-spline)

### Grid Extension

To refine the approximation, grids can be extended from $G_1$ to $G_2$ intervals:

$$\{c'_j\} = \arg\min_{c'_j} \mathbb{E}_{x \sim p(x)} \left[ \sum_{j=0}^{G_2+k-1} c'_j B'_j(x) - \sum_{i=0}^{G_1+k-1} c_i B_i(x) \right]^2$$

### Regularization

Total loss with L1 and entropy regularization:

$$\ell_{total} = \ell_{pred} + \lambda \left( \mu_1 \|\Phi\|_1 + \mu_2 S(\Phi) \right)$$

Where:
- $\|\Phi\|_1 = \sum_{i,j} |\phi_{i,j}|$ (L1 norm)
- $S(\Phi) = -\sum_{i,j} \frac{|\phi_{i,j}|}{\|\Phi\|_1} \log\frac{|\phi_{i,j}|}{\|\Phi\|_1}$ (entropy)

### Scaling Law

With B-splines of order $k=3$ (cubic):

$$\text{RMSE} \propto G^{-(k+1)} = G^{-4}$$

---

## Architecture

### Network Structure

```
KanNetwork
├── layers: Vec<KanLayer>
│   └── KanLayer
│       ├── input_size: usize
│       ├── output_size: usize
│       ├── grid_size: usize
│       ├── spline_order: usize
│       └── activations: Vec<Vec<SplineActivation>>
│           └── SplineActivation
│               ├── coeffs: Vec<f32>
│               ├── base_weight: f32
│               ├── spline_weight: f32
│               ├── grid_size: usize
│               └── spline_order: usize
├── grid_size: usize
├── grid_extend_step: usize
├── lambda_reg: f32
├── mu1: f32
└── mu2: f32
```

---

## API Reference

### Creating a KAN

```rust
use kan::network::kan::create_kan;

// Shape [2, 5, 1] with grid_size = 3
let kan = create_kan(&[2, 5, 1], 3);
```

### Forward Pass

```rust
let input = vec![0.5, 0.3];
let output = kan.forward(&input);
```

### Training

```rust
let inputs = vec![
    vec![0.1, 0.2],
    vec![0.3, 0.4],
];
let targets = vec![
    vec![0.3],
    vec![0.7],
];

// Train for 100 epochs with learning rate 0.01
kan.train(&inputs, &targets, 100, 0.01);
```

### Loss Calculation

```rust
let mse_loss = kan.mse_loss(&inputs, &targets);
let total_loss = kan.total_loss(&inputs, &targets); // with regularization
```

### Grid Extension

```rust
// Extend grid from G=3 to G=10
kan.grid_extend(10);
```

### Pruning

```rust
// Remove connections with magnitude < 0.01
kan.prune(0.01);
```

### Getting Network Info

```rust
let shape = kan.get_shape();     // vec![2, 5, 1]
let num_params = kan.num_params();
```

---

## Implementation Details

### B-Spline Implementation

The B-spline evaluation uses the de Boor's algorithm:

```rust
fn evaluate_bspline(&self, coeffs: &[f32], t: f32) -> f32 {
    let k = self.spline_order;
    let n = coeffs.len() - 1;
    
    if k == 0 {
        return coeffs[n];
    }
    
    let mut temp = coeffs.to_vec();
    
    for p in 1..=k {
        for i in 0..(n - p + 1) {
            let alpha = t / p as f32;
            temp[i] = (1.0 - alpha) * temp[i] + alpha * temp[i + 1];
        }
    }
    
    temp[0]
}
```

### SiLU Activation

$$\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$$

Derivative:

$$\text{SiLU}'(x) = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$.

### Initialization

| Parameter | Initialization |
|-----------|---------------|
| $c_i$ (spline coeffs) | $\mathcal{N}(0, 0.1^2)$ |
| $w_s$ (spline weight) | $1.0$ |
| $w_b$ (base weight) | $0.0$ (Xavier init planned) |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 3 | Number of grid intervals |
| `spline_order` | 3 | B-spline order (cubic) |
| `grid_extend_step` | 200 | Steps between grid extensions |
| `lambda_reg` | 0.01 | Regularization magnitude |
| `mu1` | 1.0 | L1 weight |
| `mu2` | 1.0 | Entropy weight |
| `prune_threshold` | 0.01 | Minimum magnitude to keep |

---

## Examples

### Regression Example

```rust
use kan::network::kan::create_kan;

fn main() {
    // Create KAN: [1, 1] for 1D regression
    let mut kan = create_kan(&[1, 1], 5);
    
    // f(x) = sin(x) * exp(x)
    let inputs: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32 / 100.0 * 2.0 * std::f32::consts::PI])
        .collect();
    let targets: Vec<Vec<f32>> = inputs.iter()
        .map(|x| vec![x[0].sin() * x[0].exp()])
        .collect();
    
    // Train
    kan.train(&inputs, &targets, 500, 0.001);
    
    // Test
    let test_input = vec![std::f32::consts::PI / 2.0];
    let output = kan.forward(&test_input);
    println!("f(π/2) ≈ {}", output[0]);
}
```

---

## References

- Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y. & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756
