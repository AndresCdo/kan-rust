pub mod kan;
// Legacy MLP compatibility surface; see `data_structures::layer`.
#[allow(
    unused_imports,
    unused_variables,
    clippy::inherent_to_string,
    clippy::module_inception,
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::should_implement_trait
)]
pub mod network;

pub use kan::{Kan, KanConfig, KanError, MAX_SERIALIZED_MODEL_BYTES, MAX_TRAINABLE_PARAMETERS};
pub use network::Network;
