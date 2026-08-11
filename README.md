# KAN in Rust

`kan` is a correctness-first Rust implementation of a fixed-grid
Kolmogorov-Arnold Network (KAN). Version 0.2.0 deliberately establishes a
small, validated core before reintroducing advanced KAN features.

## What 0.2.0 provides

- Degree-three B-spline edge activations on exterior-extended uniform knots.
- Full reverse-mode gradients through every KAN layer.
- SGD updates for spline coefficients, base weights, and spline weights.
- Deterministic initialization from an explicit seed.
- Fallible shape, batch, finite-value, and JSON validation.
- Versioned JSON persistence with inference round trips.

The layer operation follows the KAN paper:

```text
x[l + 1][j] = sum_i phi[l][j][i](x[l][i])
phi(x) = base_weight * SiLU(x) + spline_weight * sum_r coefficient[r] * B[r, 3](x)
```

For `G` grid intervals on `[a, b]`, the core uses degree `p = 3`, `G + 3`
coefficients, and `G + 7` exterior-extended uniform knots. The paper and
pykan call `p` an "order"; this crate uses the conventional term **degree**.

[docs/TECHNICAL.md](docs/TECHNICAL.md) is the core contract: implemented
versus deferred scope, the knot and gradient conventions, and the validation
rules the loader and trainer enforce.

## Installation

Build this checkout:

```bash
cargo build
```

## Usage

```rust
use kan::network::{Kan, KanConfig, KanError};

fn main() -> Result<(), KanError> {
    let config = KanConfig::new(vec![2, 5, 1], 3)?.with_seed(7);
    let mut model = Kan::try_new(config)?;

    let inputs = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let targets = vec![vec![0.3], vec![0.7]];

    model.train(&inputs, &targets, 100, 0.03)?;
    let prediction = model.forward(&[0.1, 0.2])?;
    println!("{prediction:?}");
    Ok(())
}
```

All public KAN operations validate their inputs and return `Result`.
To keep construction and loading bounded, P0 accepts at most 250,000 trainable
parameters and 8 MiB of serialized JSON source state. CLI inspection accepts
regular files only and retains an in-band bounded read.

## Persistence

```rust
use kan::network::{Kan, KanConfig, KanError};

fn round_trip() -> Result<(), KanError> {
    let model = Kan::try_new(KanConfig::new(vec![2, 1], 3)?)?;
    let encoded = model.to_json()?;
    let restored = Kan::from_json(&encoded)?;
    assert_eq!(restored.shape(), &[2, 1]);
    Ok(())
}
```

Only the versioned `kan-rust` v1 format is supported. Unversioned 0.1 KAN
JSON is rejected because its spline semantics were not canonical and cannot be
faithfully migrated.

## Binary demo

```bash
cargo run -- model.json
cargo run -- inspect model.json
```

The first command trains a deterministic `[2, 5, 1]` KAN and writes validated
JSON to the given path. The second validates and summarizes an existing model.

## Deliberately deferred

These features are **not implemented in 0.2.0**: regularization, adaptive grid
refinement, pruning, and symbolic fitting. They depend on the fixed-grid core
and are planned as separately validated follow-up work.

The older MLP, vector, and matrix APIs remain legacy compatibility code; they
are not part of the KAN correctness guarantee and will be handled separately.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
```

## References

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v5)
- [pykan reference implementation](https://github.com/KindXiaoming/pykan)
- [SciPy B-spline reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)

## License

[MIT](LICENSE)
