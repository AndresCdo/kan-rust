pub fn relu(x: f64) -> f64 {
  if x > 0.0 { x } else { 0.0 }
}

pub fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}
