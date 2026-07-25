// Legacy MLP compatibility surface. Its behavior is intentionally unchanged in
// the KAN-first 0.2.0 milestone and will be remediated separately.
#[allow(
    unused_imports,
    unused_variables,
    clippy::let_and_return,
    clippy::should_implement_trait
)]
pub mod layer;
#[allow(
    unused_imports,
    unused_variables,
    mismatched_lifetime_syntaxes,
    clippy::inherent_to_string,
    clippy::needless_range_loop,
    clippy::redundant_closure
)]
pub mod matrix;
pub mod spline;
#[allow(unused_parens, clippy::inherent_to_string)]
pub mod vector;

pub use layer::Layer;
pub use matrix::Matrix;
pub use vector::Vector;
