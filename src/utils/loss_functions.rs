pub fn mean_squared_error(predictions: &[f64], targets: &[f64]) -> f64 {
  predictions.iter().zip(targets.iter())
      .map(|(p, t)| (p - t).powi(2))
      .sum::<f64>() / predictions.len() as f64
}
