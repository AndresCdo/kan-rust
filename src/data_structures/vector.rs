use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub elements: Vec<f64>,
}

impl Vector {
    pub fn new(elements: Vec<f64>) -> Self {
        Vector { elements }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        self.elements.iter().zip(&other.elements).map(|(a, b)| a * b).sum()
    }

    pub fn add(&self, other: &Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a + b).collect()
        )
    }

    pub fn add_scalar(&self, scalar: f64) -> Vector {
        Vector::new(self.elements.iter().map(|a| a + scalar).collect())
    }

    pub fn scalar_multiply(&self, scalar: f64) -> Vector {
        Vector::new(self.elements.iter().map(|a| a * scalar).collect())
    }

    pub fn subtract(&self, other: &Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a - b).collect()
        )
    }

    pub fn elementwise_multiply(&self, other: &Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a * b).collect()
        )
    }

    pub fn elementwise_divide(&self, other: &Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a / b).collect()
        )
    }

    pub fn elementwise_sqrt(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| x.sqrt()).collect())
    }

    pub fn sum(&self) -> f64 {
        self.elements.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.len() as f64
    }

    pub fn sigmoid(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
    }

    pub fn sigmoid_derivative(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| x * (1.0 - x)).collect())
    }

    pub fn relu(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| x.max(0.0)).collect())
    }

    pub fn relu_derivative(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect())
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.elements.clone()
    }

    pub fn from_vec(vec: Vec<f64>) -> Vector {
        Vector::new(vec)
    }

    pub fn to_string(&self) -> String {
        format!("{:?}", self.elements)
    }

    pub fn from_string(s: &str) -> Result<Vector, std::num::ParseFloatError> {
        let elements: Result<Vec<f64>, _> = s.split(", ").map(str::parse).collect();
        elements.map(Vector::new)
    }

    pub fn random(size: usize) -> Vector {
        // Non NULL random number generator
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Vector::new((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect())
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Vector {
        Vector::new(self.elements.iter().map(|&x| f(x)).collect())
    }

    pub fn map_with_index(&self, f: impl Fn(f64, usize) -> f64) -> Vector {
        Vector::new(self.elements.iter().enumerate().map(|(i, &x)| f(x, i)).collect())
    }

    pub fn map_with_vector(&self, other: &Vector, f: impl Fn(f64, f64) -> f64) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(&a, &b)| f(a, b)).collect()
        )
    }

    pub fn map_with_vector_index(&self, other: &Vector, f: impl Fn(f64, f64, usize) -> f64) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).enumerate().map(|(i, (&a, &b))| f(a, b, i)).collect()
        )
    }

    pub fn unwrap(&self) -> Result<Vector, String> {
        if self.elements.iter().all(|&x| x.is_finite()) {
            Ok(self.clone())
        } else {
            Err("Vector contains non-finite elements".to_string())
        }
    }
}
