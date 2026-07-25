//! A validated, fixed-grid cubic Kolmogorov-Arnold Network implementation.
//!
//! The P0 model uses degree-three B-splines on exterior-extended uniform knots,
//! matching the grid convention in the KAN paper and pykan. Grid refinement,
//! regularization, pruning, and symbolic fitting are intentionally deferred.

use crate::data_structures::spline::{
    coefficient_count, silu, silu_derivative, SplineActivation, SPLINE_DEGREE,
};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

const FORMAT_NAME: &str = "kan-rust";
const FORMAT_VERSION: u32 = 1;
const DEFAULT_DOMAIN: [f32; 2] = [-1.0, 1.0];
const MAX_JSON_NODES: usize = 1_000_000;

/// Largest accepted number of trainable scalar parameters.
pub const MAX_TRAINABLE_PARAMETERS: usize = 250_000;

/// Largest accepted UTF-8 JSON source-state document, in bytes.
pub const MAX_SERIALIZED_MODEL_BYTES: usize = 8 * 1024 * 1024;

/// Configuration used to construct a fixed-grid KAN.
#[derive(Clone, Debug, PartialEq)]
pub struct KanConfig {
    shape: Vec<usize>,
    grid_intervals: usize,
    domain: [f32; 2],
    seed: u64,
}

impl KanConfig {
    /// Creates a configuration with the default spline domain `[-1, 1]`.
    pub fn new(shape: Vec<usize>, grid_intervals: usize) -> Result<Self, KanError> {
        let config = Self {
            shape,
            grid_intervals,
            domain: DEFAULT_DOMAIN,
            seed: 0,
        };
        config.validate()?;
        Ok(config)
    }

    /// Changes the common spline domain after validating its finite bounds.
    pub fn with_domain(mut self, domain: [f32; 2]) -> Result<Self, KanError> {
        self.domain = domain;
        self.validate()?;
        Ok(self)
    }

    /// Sets the deterministic initialization seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn validate(&self) -> Result<(), KanError> {
        validate_shape(&self.shape)?;
        if self.grid_intervals == 0 {
            return Err(KanError::InvalidGridIntervals);
        }
        if self.grid_intervals > isize::MAX as usize - (2 * SPLINE_DEGREE + 1) {
            return Err(KanError::InvalidGridIntervals);
        }
        validate_domain(self.domain)?;

        let coefficients_per_edge = self
            .grid_intervals
            .checked_add(SPLINE_DEGREE)
            .ok_or(KanError::InvalidGridIntervals)?;
        self.grid_intervals
            .checked_add(2 * SPLINE_DEGREE + 1)
            .ok_or(KanError::InvalidGridIntervals)?;
        let interval_width = (self.domain[1] - self.domain[0]) / self.grid_intervals as f32;
        let extended_min = self.domain[0] - SPLINE_DEGREE as f32 * interval_width;
        let extended_max = self.domain[1] + SPLINE_DEGREE as f32 * interval_width;
        if !interval_width.is_finite()
            || interval_width <= 0.0
            || !extended_min.is_finite()
            || !extended_max.is_finite()
            || self.domain[0] + interval_width <= self.domain[0]
            || self.domain[1] - interval_width >= self.domain[1]
            || extended_min >= self.domain[0]
            || extended_max <= self.domain[1]
        {
            return Err(KanError::InvalidDomain);
        }

        let parameters_per_edge = coefficients_per_edge
            .checked_add(2)
            .ok_or(KanError::InvalidGridIntervals)?;
        let mut total_parameters = 0_usize;
        for widths in self.shape.windows(2) {
            let edge_count = widths[0].checked_mul(widths[1]).ok_or_else(|| {
                KanError::InvalidShape("layer edge count overflows usize".to_owned())
            })?;
            let layer_parameters =
                edge_count.checked_mul(parameters_per_edge).ok_or_else(|| {
                    KanError::InvalidShape("layer parameter count overflows usize".to_owned())
                })?;
            total_parameters = total_parameters
                .checked_add(layer_parameters)
                .ok_or_else(|| {
                    KanError::InvalidShape("network parameter count overflows usize".to_owned())
                })?;
        }
        if total_parameters > MAX_TRAINABLE_PARAMETERS {
            return Err(KanError::ModelTooLarge {
                parameters: total_parameters,
                limit: MAX_TRAINABLE_PARAMETERS,
            });
        }
        Ok(())
    }
}

/// Errors returned when KAN configuration, data, or persisted state is invalid.
#[derive(Debug, Error)]
pub enum KanError {
    /// The network topology cannot form one or more non-empty layers.
    #[error("invalid KAN shape: {0}")]
    InvalidShape(String),
    /// The spline grid interval count is zero or cannot be represented safely.
    #[error("grid interval count is zero or too large")]
    InvalidGridIntervals,
    /// The spline domain is non-finite or does not satisfy `min < max`.
    #[error("domain must define a finite, representable uniform grid with min < max")]
    InvalidDomain,
    /// An input has a width different from the model input width.
    #[error("input width mismatch: expected {expected}, got {actual}")]
    InputWidthMismatch { expected: usize, actual: usize },
    /// A target has a width different from the model output width.
    #[error("target width mismatch: expected {expected}, got {actual}")]
    TargetWidthMismatch { expected: usize, actual: usize },
    /// Inputs and targets contain a different number of samples.
    #[error("batch size mismatch: {inputs} inputs, {targets} targets")]
    BatchSizeMismatch { inputs: usize, targets: usize },
    /// A batch contains no samples.
    #[error("batches must not be empty")]
    EmptyBatch,
    /// An input contains `NaN` or infinity.
    #[error("input contains a non-finite value")]
    NonFiniteInput,
    /// A target contains `NaN` or infinity.
    #[error("target contains a non-finite value")]
    NonFiniteTarget,
    /// A model parameter contains `NaN` or infinity.
    #[error("model contains a non-finite parameter")]
    NonFiniteParameter,
    /// The learning rate is not finite and strictly positive.
    #[error("learning rate must be finite and positive")]
    InvalidLearningRate,
    /// The validated topology exceeds the fixed P0 resource budget.
    #[error("model has {parameters} parameters, exceeding the limit of {limit}")]
    ModelTooLarge { parameters: usize, limit: usize },
    /// The serialized source-state document exceeds the fixed P0 input budget.
    #[error("serialized model has {bytes} bytes, exceeding the limit of {limit}")]
    SerializedModelTooLarge { bytes: usize, limit: usize },
    /// The serialized source state has too many JSON values to parse safely.
    #[error("serialized model exceeds the JSON complexity limit of {limit} values")]
    SerializedModelTooComplex { limit: usize },
    /// A finite operation overflowed or otherwise produced a non-finite result.
    #[error("numerical failure during {0}")]
    NumericalFailure(&'static str),
    /// The JSON document does not use the P0 versioned envelope.
    #[error("legacy unversioned KAN models are unsupported")]
    LegacyUnversionedModel,
    /// The JSON document uses an unsupported format version.
    #[error("unsupported KAN model format version {0}")]
    UnsupportedFormatVersion(u32),
    /// The serialized model violates a structural invariant.
    #[error("malformed KAN model: {0}")]
    MalformedModel(String),
    /// JSON serialization or parsing failed.
    #[error("KAN JSON error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// A fixed-grid, fully connected KAN with cubic edge activations.
#[derive(Clone, Debug)]
pub struct Kan {
    config: KanConfig,
    layers: Vec<KanLayer>,
}

#[derive(Clone, Debug)]
struct KanLayer {
    input_width: usize,
    output_width: usize,
    edges: Vec<SplineActivation>,
}

#[derive(Clone, Debug)]
struct ForwardCache {
    node_values: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct LayerGradient {
    base_weights: Vec<f32>,
    spline_weights: Vec<f32>,
    coefficients: Vec<Vec<f32>>,
}

impl Kan {
    /// Constructs a validated KAN with deterministic parameter initialization.
    pub fn try_new(config: KanConfig) -> Result<Self, KanError> {
        config.validate()?;
        let mut random = SplitMix64::new(config.seed);
        let mut layers = Vec::with_capacity(config.shape.len() - 1);

        for widths in config.shape.windows(2) {
            layers.push(KanLayer::random(
                widths[0],
                widths[1],
                config.grid_intervals,
                config.domain,
                &mut random,
            ));
        }

        Ok(Self { config, layers })
    }

    /// Returns the immutable network topology.
    pub fn shape(&self) -> &[usize] {
        &self.config.shape
    }

    /// Returns the number of trainable scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.edges.len() * (coefficient_count(self.config.grid_intervals) + 2))
            .sum()
    }

    /// Evaluates the KAN for one finite input of the configured width.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, KanError> {
        self.validate_input(input)?;
        self.forward_cached(input).map(|(output, _)| output)
    }

    /// Evaluates the KAN for a batch of finite inputs.
    pub fn predict_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, KanError> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Returns mean squared error reduced across samples and output coordinates.
    pub fn mse_loss(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<f32, KanError> {
        self.validate_dataset(inputs, targets)?;
        let mut sum = 0.0;
        for (input, target) in inputs.iter().zip(targets) {
            let output = self.forward(input)?;
            for (prediction, expected) in output.iter().zip(target) {
                let squared_error = (prediction - expected).powi(2);
                if !squared_error.is_finite() {
                    return Err(KanError::NumericalFailure("MSE calculation"));
                }
                sum += squared_error;
                if !sum.is_finite() {
                    return Err(KanError::NumericalFailure("MSE accumulation"));
                }
            }
        }
        Ok(sum / (inputs.len() * self.config.shape[self.config.shape.len() - 1]) as f32)
    }

    /// Applies one full-batch SGD step to every edge parameter in every layer.
    pub fn train_step(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        learning_rate: f32,
    ) -> Result<(), KanError> {
        self.validate_dataset(inputs, targets)?;
        validate_learning_rate(learning_rate)?;

        let mut gradients: Vec<LayerGradient> = self
            .layers
            .iter()
            .map(|layer| LayerGradient::zeros(layer, self.config.grid_intervals))
            .collect();
        let output_width = self.config.shape[self.config.shape.len() - 1];
        let loss_scale = 2.0 / (inputs.len() * output_width) as f32;

        for (input, target) in inputs.iter().zip(targets) {
            let (output, cache) = self.forward_cached(input)?;
            let output_gradient: Vec<f32> = output
                .iter()
                .zip(target)
                .map(|(prediction, expected)| loss_scale * (prediction - expected))
                .collect();
            self.backward_accumulate(&cache, output_gradient, &mut gradients);
        }

        let mut updated_layers = self.layers.clone();
        for (layer, gradient) in updated_layers.iter_mut().zip(gradients) {
            for (edge_index, edge) in layer.edges.iter_mut().enumerate() {
                edge.base_weight -= learning_rate * gradient.base_weights[edge_index];
                edge.spline_weight -= learning_rate * gradient.spline_weights[edge_index];
                for (coefficient, coefficient_gradient) in edge
                    .coefficients
                    .iter_mut()
                    .zip(&gradient.coefficients[edge_index])
                {
                    *coefficient -= learning_rate * coefficient_gradient;
                }
            }
        }
        if !parameters_are_finite(&updated_layers) {
            return Err(KanError::NumericalFailure("parameter update"));
        }
        self.layers = updated_layers;

        Ok(())
    }

    /// Runs deterministic full-batch SGD for `epochs` steps.
    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        epochs: usize,
        learning_rate: f32,
    ) -> Result<(), KanError> {
        self.validate_dataset(inputs, targets)?;
        validate_learning_rate(learning_rate)?;
        for _ in 0..epochs {
            self.train_step(inputs, targets, learning_rate)?;
        }
        Ok(())
    }

    /// Serializes the versioned source state required to reproduce inference.
    pub fn to_json(&self) -> Result<String, KanError> {
        if !parameters_are_finite(&self.layers) {
            return Err(KanError::NonFiniteParameter);
        }
        let wire = WireDocument {
            format: FORMAT_NAME.to_owned(),
            format_version: FORMAT_VERSION,
            model: WireModel {
                shape: self.config.shape.clone(),
                spline: WireSpline {
                    degree: 3,
                    grid_intervals: self.config.grid_intervals,
                    domain: self.config.domain,
                },
                layers: self.layers.iter().map(KanLayer::to_wire).collect(),
            },
        };
        let encoded = serde_json::to_string(&wire)?;
        validate_serialized_size(encoded.len())?;
        Ok(encoded)
    }

    /// Loads and validates a P0 versioned source-state JSON document.
    pub fn from_json(encoded: &str) -> Result<Self, KanError> {
        validate_serialized_size(encoded.len())?;
        preflight_json(encoded, MAX_JSON_NODES)?;
        let wire: WireDocument = match serde_json::from_str(encoded) {
            Ok(wire) => wire,
            Err(_) if !encoded.contains("\"format_version\"") => {
                return Err(KanError::LegacyUnversionedModel)
            }
            Err(error) => return Err(KanError::Serialization(error)),
        };
        if wire.format != FORMAT_NAME {
            return Err(KanError::MalformedModel(
                "unexpected format name".to_owned(),
            ));
        }
        if wire.format_version != FORMAT_VERSION {
            return Err(KanError::UnsupportedFormatVersion(wire.format_version));
        }
        if wire.model.spline.degree != 3 {
            return Err(KanError::MalformedModel(
                "only degree-three splines are supported".to_owned(),
            ));
        }

        let config = KanConfig::new(wire.model.shape, wire.model.spline.grid_intervals)?
            .with_domain(wire.model.spline.domain)?;
        if wire.model.layers.len() != config.shape.len() - 1 {
            return Err(KanError::MalformedModel(
                "layer count does not match topology".to_owned(),
            ));
        }

        let mut layers = Vec::with_capacity(wire.model.layers.len());
        for (index, layer) in wire.model.layers.into_iter().enumerate() {
            layers.push(KanLayer::from_wire(
                layer,
                config.shape[index],
                config.shape[index + 1],
                config.grid_intervals,
                config.domain,
            )?);
        }

        Ok(Self { config, layers })
    }

    fn forward_cached(&self, input: &[f32]) -> Result<(Vec<f32>, ForwardCache), KanError> {
        let mut node_values = Vec::with_capacity(self.layers.len() + 1);
        node_values.push(input.to_vec());
        for layer in &self.layers {
            let current_index = node_values.len() - 1;
            let next = layer.forward(&node_values[current_index]);
            if next.iter().any(|value| !value.is_finite()) {
                return Err(KanError::NumericalFailure("forward propagation"));
            }
            node_values.push(next);
        }
        let output = node_values[node_values.len() - 1].clone();
        Ok((output, ForwardCache { node_values }))
    }

    fn backward_accumulate(
        &self,
        cache: &ForwardCache,
        mut output_gradient: Vec<f32>,
        gradients: &mut [LayerGradient],
    ) {
        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let input = &cache.node_values[layer_index];
            let mut input_gradient = vec![0.0; layer.input_width];

            for (output_index, upstream) in output_gradient.iter().copied().enumerate() {
                for input_index in 0..layer.input_width {
                    let edge_index = layer.edge_index(output_index, input_index);
                    let edge = &layer.edges[edge_index];
                    let basis = edge.basis(input[input_index]);
                    let (spline, spline_derivative) = edge.spline_value_and_derivative(&basis);

                    gradients[layer_index].base_weights[edge_index] +=
                        upstream * silu(input[input_index]);
                    gradients[layer_index].spline_weights[edge_index] += upstream * spline;
                    for (gradient, basis_value) in gradients[layer_index].coefficients[edge_index]
                        .iter_mut()
                        .zip(&basis.values)
                    {
                        *gradient += upstream * edge.spline_weight * basis_value;
                    }

                    input_gradient[input_index] += upstream
                        * (edge.base_weight * silu_derivative(input[input_index])
                            + edge.spline_weight * spline_derivative);
                }
            }
            output_gradient = input_gradient;
        }
    }

    fn validate_input(&self, input: &[f32]) -> Result<(), KanError> {
        let expected = self.config.shape[0];
        if input.len() != expected {
            return Err(KanError::InputWidthMismatch {
                expected,
                actual: input.len(),
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(KanError::NonFiniteInput);
        }
        Ok(())
    }

    fn validate_dataset(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<(), KanError> {
        if inputs.len() != targets.len() {
            return Err(KanError::BatchSizeMismatch {
                inputs: inputs.len(),
                targets: targets.len(),
            });
        }
        if inputs.is_empty() {
            return Err(KanError::EmptyBatch);
        }
        let expected_target_width = self.config.shape[self.config.shape.len() - 1];
        for input in inputs {
            self.validate_input(input)?;
        }
        for target in targets {
            if target.len() != expected_target_width {
                return Err(KanError::TargetWidthMismatch {
                    expected: expected_target_width,
                    actual: target.len(),
                });
            }
            if target.iter().any(|value| !value.is_finite()) {
                return Err(KanError::NonFiniteTarget);
            }
        }
        Ok(())
    }
}

impl KanLayer {
    fn random(
        input_width: usize,
        output_width: usize,
        grid_intervals: usize,
        domain: [f32; 2],
        random: &mut SplitMix64,
    ) -> Self {
        let mut edges = Vec::with_capacity(input_width * output_width);
        for _ in 0..(input_width * output_width) {
            let base_weight = random.signed_uniform() / (input_width as f32).sqrt();
            let spline_weight = 1.0;
            let coefficients = (0..coefficient_count(grid_intervals))
                .map(|_| 0.01 * random.signed_uniform())
                .collect();
            edges.push(SplineActivation::new(
                grid_intervals,
                domain,
                base_weight,
                spline_weight,
                coefficients,
            ));
        }
        Self {
            input_width,
            output_width,
            edges,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_width];
        for (output_index, node) in output.iter_mut().enumerate() {
            for (input_index, input_value) in input.iter().copied().enumerate() {
                let (value, _) =
                    self.edges[self.edge_index(output_index, input_index)].evaluate(input_value);
                *node += value;
            }
        }
        output
    }

    fn edge_index(&self, output_index: usize, input_index: usize) -> usize {
        output_index * self.input_width + input_index
    }

    fn to_wire(&self) -> WireLayer {
        let mut base_weights = vec![vec![0.0; self.input_width]; self.output_width];
        let mut spline_weights = vec![vec![0.0; self.input_width]; self.output_width];
        let mut coefficients =
            vec![
                vec![vec![0.0; self.edges[0].coefficients.len()]; self.input_width];
                self.output_width
            ];
        for output_index in 0..self.output_width {
            for input_index in 0..self.input_width {
                let edge = &self.edges[self.edge_index(output_index, input_index)];
                base_weights[output_index][input_index] = edge.base_weight;
                spline_weights[output_index][input_index] = edge.spline_weight;
                coefficients[output_index][input_index] = edge.coefficients.clone();
            }
        }
        WireLayer {
            base_weights,
            spline_weights,
            coefficients,
        }
    }

    fn from_wire(
        wire: WireLayer,
        input_width: usize,
        output_width: usize,
        grid_intervals: usize,
        domain: [f32; 2],
    ) -> Result<Self, KanError> {
        validate_matrix(
            &wire.base_weights,
            output_width,
            input_width,
            "base_weights",
        )?;
        validate_matrix(
            &wire.spline_weights,
            output_width,
            input_width,
            "spline_weights",
        )?;
        if wire.coefficients.len() != output_width {
            return Err(KanError::MalformedModel(
                "coefficient output dimension mismatch".to_owned(),
            ));
        }
        let coefficient_len = coefficient_count(grid_intervals);
        let mut edges = Vec::with_capacity(input_width * output_width);
        for output_index in 0..output_width {
            if wire.coefficients[output_index].len() != input_width {
                return Err(KanError::MalformedModel(
                    "coefficient input dimension mismatch".to_owned(),
                ));
            }
            for input_index in 0..input_width {
                let coefficients = &wire.coefficients[output_index][input_index];
                if coefficients.len() != coefficient_len {
                    return Err(KanError::MalformedModel(
                        "coefficient count does not match grid".to_owned(),
                    ));
                }
                let base_weight = wire.base_weights[output_index][input_index];
                let spline_weight = wire.spline_weights[output_index][input_index];
                if !base_weight.is_finite()
                    || !spline_weight.is_finite()
                    || coefficients.iter().any(|value| !value.is_finite())
                {
                    return Err(KanError::NonFiniteParameter);
                }
                edges.push(SplineActivation::new(
                    grid_intervals,
                    domain,
                    base_weight,
                    spline_weight,
                    coefficients.clone(),
                ));
            }
        }
        Ok(Self {
            input_width,
            output_width,
            edges,
        })
    }
}

impl LayerGradient {
    fn zeros(layer: &KanLayer, grid_intervals: usize) -> Self {
        Self {
            base_weights: vec![0.0; layer.edges.len()],
            spline_weights: vec![0.0; layer.edges.len()],
            coefficients: vec![vec![0.0; coefficient_count(grid_intervals)]; layer.edges.len()],
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDocument {
    format: String,
    format_version: u32,
    model: WireModel,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireModel {
    shape: Vec<usize>,
    spline: WireSpline,
    layers: Vec<WireLayer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSpline {
    degree: usize,
    grid_intervals: usize,
    domain: [f32; 2],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLayer {
    base_weights: Vec<Vec<f32>>,
    spline_weights: Vec<Vec<f32>>,
    coefficients: Vec<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn signed_uniform(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / ((1_u64 << 24) - 1) as f32;
        2.0 * unit - 1.0
    }
}

fn validate_shape(shape: &[usize]) -> Result<(), KanError> {
    if shape.len() < 2 || shape.contains(&0) {
        return Err(KanError::InvalidShape(
            "at least two non-zero layer widths are required".to_owned(),
        ));
    }
    for widths in shape.windows(2) {
        widths[0]
            .checked_mul(widths[1])
            .ok_or_else(|| KanError::InvalidShape("layer edge count overflows usize".to_owned()))?;
    }
    Ok(())
}

fn validate_domain(domain: [f32; 2]) -> Result<(), KanError> {
    if !domain[0].is_finite() || !domain[1].is_finite() || domain[0] >= domain[1] {
        return Err(KanError::InvalidDomain);
    }
    Ok(())
}

fn validate_learning_rate(learning_rate: f32) -> Result<(), KanError> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(KanError::InvalidLearningRate);
    }
    Ok(())
}

fn parameters_are_finite(layers: &[KanLayer]) -> bool {
    layers.iter().all(|layer| {
        layer.edges.iter().all(|edge| {
            edge.base_weight.is_finite()
                && edge.spline_weight.is_finite()
                && edge.coefficients.iter().all(|value| value.is_finite())
        })
    })
}

fn validate_matrix(
    matrix: &[Vec<f32>],
    expected_rows: usize,
    expected_columns: usize,
    name: &str,
) -> Result<(), KanError> {
    if matrix.len() != expected_rows || matrix.iter().any(|row| row.len() != expected_columns) {
        return Err(KanError::MalformedModel(format!(
            "{name} shape does not match topology"
        )));
    }
    Ok(())
}

fn validate_serialized_size(bytes: usize) -> Result<(), KanError> {
    if bytes > MAX_SERIALIZED_MODEL_BYTES {
        return Err(KanError::SerializedModelTooLarge {
            bytes,
            limit: MAX_SERIALIZED_MODEL_BYTES,
        });
    }
    Ok(())
}

fn preflight_json(encoded: &str, node_limit: usize) -> Result<(), KanError> {
    let mut budget = JsonBudget {
        remaining: node_limit,
        exceeded: false,
    };
    let mut deserializer = serde_json::Deserializer::from_str(encoded);
    let result = JsonBudgetSeed {
        budget: &mut budget,
    }
    .deserialize(&mut deserializer);

    if budget.exceeded {
        return Err(KanError::SerializedModelTooComplex { limit: node_limit });
    }
    result?;
    deserializer.end()?;
    Ok(())
}

struct JsonBudget {
    remaining: usize,
    exceeded: bool,
}

impl JsonBudget {
    fn consume<E: de::Error>(&mut self) -> Result<(), E> {
        if self.remaining == 0 {
            self.exceeded = true;
            return Err(E::custom("JSON value limit exceeded"));
        }
        self.remaining -= 1;
        Ok(())
    }
}

struct JsonBudgetSeed<'a> {
    budget: &'a mut JsonBudget,
}

impl<'de> DeserializeSeed<'de> for JsonBudgetSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        self.budget.consume::<D::Error>()?;
        deserializer.deserialize_any(JsonBudgetVisitor {
            budget: self.budget,
        })
    }
}

struct JsonBudgetVisitor<'a> {
    budget: &'a mut JsonBudget,
}

impl<'de> Visitor<'de> for JsonBudgetVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value within the configured complexity limit")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        JsonBudgetSeed {
            budget: self.budget,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence
            .next_element_seed(JsonBudgetSeed {
                budget: self.budget,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while map
            .next_key_seed(JsonBudgetSeed {
                budget: self.budget,
            })?
            .is_some()
        {
            map.next_value_seed(JsonBudgetSeed {
                budget: self.budget,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preflight_json, validate_serialized_size, Kan, KanConfig, KanError, MAX_JSON_NODES,
        MAX_SERIALIZED_MODEL_BYTES,
    };

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    #[test]
    fn serialized_resource_limits_are_enforced_before_typed_deserialization() {
        assert!(validate_serialized_size(MAX_SERIALIZED_MODEL_BYTES).is_ok());
        assert!(matches!(
            validate_serialized_size(MAX_SERIALIZED_MODEL_BYTES + 1),
            Err(KanError::SerializedModelTooLarge { .. })
        ));
        assert!(preflight_json(r#"{"a":[1,2]}"#, MAX_JSON_NODES).is_ok());
        assert!(matches!(
            preflight_json("[0,0,0]", 3),
            Err(KanError::SerializedModelTooComplex { limit: 3 })
        ));
    }

    #[test]
    fn configured_json_node_budget_accepts_its_exact_boundary() {
        let element_count = MAX_JSON_NODES - 1;
        let mut exact = String::with_capacity(element_count * 2 + 1);
        exact.push('[');
        for index in 0..element_count {
            if index != 0 {
                exact.push(',');
            }
            exact.push('0');
        }
        exact.push(']');

        assert!(preflight_json(&exact, MAX_JSON_NODES).is_ok());
        let mut over_limit = exact;
        over_limit.pop();
        over_limit.push_str(",0]");
        assert!(matches!(
            preflight_json(&over_limit, MAX_JSON_NODES),
            Err(KanError::SerializedModelTooComplex {
                limit: MAX_JSON_NODES
            })
        ));
    }

    #[test]
    fn finite_differences_match_all_edge_parameter_families() {
        let mut model =
            Kan::try_new(KanConfig::new(vec![2, 2, 1], 3).unwrap().with_seed(4)).unwrap();
        let input = vec![vec![0.17, -0.29]];
        let target = vec![vec![-0.23]];
        let epsilon = 1.0e-3;

        let (_, cache) = model.forward_cached(&input[0]).unwrap();
        let output = model.forward(&input[0]).unwrap();
        let mut gradients = model
            .layers
            .iter()
            .map(|layer| super::LayerGradient::zeros(layer, model.config.grid_intervals))
            .collect::<Vec<_>>();
        let upstream = vec![2.0 * (output[0] - target[0][0])];
        model.backward_accumulate(&cache, upstream, &mut gradients);

        for (layer_index, layer_gradient) in gradients.iter().enumerate() {
            for (edge_index, coefficient_gradients) in
                layer_gradient.coefficients.iter().enumerate()
            {
                for (coefficient_index, analytic_gradient) in
                    coefficient_gradients.iter().copied().enumerate()
                {
                    let original =
                        model.layers[layer_index].edges[edge_index].coefficients[coefficient_index];
                    model.layers[layer_index].edges[edge_index].coefficients[coefficient_index] =
                        original + epsilon;
                    let plus = model.mse_loss(&input, &target).unwrap();
                    model.layers[layer_index].edges[edge_index].coefficients[coefficient_index] =
                        original - epsilon;
                    let minus = model.mse_loss(&input, &target).unwrap();
                    model.layers[layer_index].edges[edge_index].coefficients[coefficient_index] =
                        original;
                    close(analytic_gradient, (plus - minus) / (2.0 * epsilon), 3.0e-3);
                }

                let original = model.layers[layer_index].edges[edge_index].base_weight;
                model.layers[layer_index].edges[edge_index].base_weight = original + epsilon;
                let plus = model.mse_loss(&input, &target).unwrap();
                model.layers[layer_index].edges[edge_index].base_weight = original - epsilon;
                let minus = model.mse_loss(&input, &target).unwrap();
                model.layers[layer_index].edges[edge_index].base_weight = original;
                close(
                    layer_gradient.base_weights[edge_index],
                    (plus - minus) / (2.0 * epsilon),
                    3.0e-3,
                );

                let original = model.layers[layer_index].edges[edge_index].spline_weight;
                model.layers[layer_index].edges[edge_index].spline_weight = original + epsilon;
                let plus = model.mse_loss(&input, &target).unwrap();
                model.layers[layer_index].edges[edge_index].spline_weight = original - epsilon;
                let minus = model.mse_loss(&input, &target).unwrap();
                model.layers[layer_index].edges[edge_index].spline_weight = original;
                close(
                    layer_gradient.spline_weights[edge_index],
                    (plus - minus) / (2.0 * epsilon),
                    3.0e-3,
                );
            }
        }
    }

    #[test]
    fn one_training_step_updates_every_parameter_family_in_every_layer() {
        let mut model =
            Kan::try_new(KanConfig::new(vec![1, 1, 1], 3).unwrap().with_seed(8)).unwrap();
        let before = model.layers.clone();
        model.train_step(&[vec![0.2]], &[vec![0.9]], 0.05).unwrap();
        for (before_layer, after_layer) in before.iter().zip(&model.layers) {
            for (before_edge, after_edge) in before_layer.edges.iter().zip(&after_layer.edges) {
                assert_ne!(before_edge.base_weight, after_edge.base_weight);
                assert_ne!(before_edge.spline_weight, after_edge.spline_weight);
                assert!(before_edge
                    .coefficients
                    .iter()
                    .zip(&after_edge.coefficients)
                    .any(|(before, after)| before != after));
            }
        }
    }
}
