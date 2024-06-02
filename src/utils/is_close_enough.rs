pub fn is_close_enough(output: &[f64], expected: &[f64]) -> bool {
    let epsilon = 0.0001;
    output.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < epsilon)
}