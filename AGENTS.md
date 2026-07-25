# AGENTS.md - Documentation for AI Agents

This file contains instructions for AI agents working on this project.

## Project Information

- **Name**: KAN (Kolmogorov-Arnold Networks) Rust Implementation
- **Language**: Rust
- **Type**: Library + Binary
- **Repository**: https://github.com/AndresCdo/kan-rust

## Project Structure

```
kan-rust/
├── src/
│   ├── bin/kan.rs           # Binary entry point
│   ├── data_structures/     # Core data structures
│   │   ├── vector.rs       # Vector operations
│   │   ├── matrix.rs      # Matrix operations
│   │   ├── layer.rs       # MLP layer (legacy)
│   │   └── spline.rs      # B-spline implementation
│   ├── network/            # Network implementations
│   │   ├── network.rs     # MLP network (legacy)
│   │   └── kan.rs         # KAN implementation ✓
│   ├── utils/             # Utilities
│   └── tests/             # Unit tests
├── docs/
│   └── TECHNICAL.md       # Technical documentation
├── Cargo.toml
└── README.md
```

## Development Commands

### Compilation

```bash
# Development
cargo build

# Release
cargo build --release

# Verify without compiling
cargo check
```

### Testing

```bash
# All targets
cargo test --locked --all-targets

# Library tests
cargo test --lib

# Tests with output
cargo test -- --nocapture

# Coverage
cargo tarpaulin
```

### Linting

```bash
# Auto-fix
cargo fix

# Clippy
cargo clippy --locked --all-targets -- -D warnings

# Formatting gate
cargo fmt --all -- --check
```

### Documentation

```bash
# Generate docs
cargo doc --open

# Generate docs with warnings denied
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
```

## KAN Architecture

The project implements **Kolmogorov-Arnold Networks** according to the paper:

- **Paper**: [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v5)
- **Original Python Repo**: https://github.com/KindXiaoming/pykan

### Implemented Features

| Feature | Status | File |
|---------|--------|------|
| B-spline activation | ✅ | `spline.rs` |
| SiLU base function | ✅ | `spline.rs` |
| Forward propagation | ✅ | `kan.rs` |
| Full multilayer backpropagation | ✅ | `kan.rs` |
| Deterministic initialization | ✅ | `kan.rs` |
| Versioned validated persistence | ✅ | `kan.rs` |
| Grid extension | Deferred | — |
| L1/entropy regularization | Deferred | — |
| Node pruning | Deferred | — |
| Symbolic fixing | Deferred | — |

## Main API

### Create a validated KAN

```rust
use kan::network::{Kan, KanConfig};

let config = KanConfig::new(vec![2, 5, 1], 3)?.with_seed(7);
let mut kan = Kan::try_new(config)?;
```

### Forward Pass

```rust
let input = vec![0.5, 0.3];
let output = kan.forward(&input)?;
```

### Training

```rust
kan.train(&inputs, &targets, epochs, learning_rate)?;
```

### Loss Functions

```rust
let mse = kan.mse_loss(&inputs, &targets)?;
```

## Code Standards

- Rust 2021 edition
- Warnings as errors in CI
- Unit tests for new features
- Documentation for public functions
- snake_case naming
- Concrete types preferred over inference

## Testing Requirements

Before commit:

1. `cargo test` must pass
2. `cargo clippy` without warnings
3. `cargo fmt` formatted
4. Documentation updated if API changes

## Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
rand = "0.8.6"
indicatif = "0.18"
```

## Notes for Agents

- Legacy code (MLP) is in `layer.rs` and `network.rs`
- New KAN implementation is in `spline.rs` and `kan.rs`
- KAN mathematical tests are inline; public contract tests are in `tests/kan_core.rs`.
- KAN fields are intentionally private; persisted models must pass `Kan::from_json` validation.
- The legacy MLP/vector/matrix surface is outside the 0.2.0 KAN correctness guarantee.
- Regularization, grid refinement, pruning, and symbolic fitting require separate designs and tests before reintroduction.
