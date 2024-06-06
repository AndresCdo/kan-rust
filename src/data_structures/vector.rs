use std::ops::{MulAssign, Sub};
use crate::data_structures::Matrix;
use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub elements: Vec<f32>,
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, other: Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(other.elements).map(|(a, b)| a - b).collect()
        )
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, scalar: f32) {
        self.elements.iter_mut().for_each(|x| *x *= scalar);
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, other: Vector) {
        self.elements.iter_mut().zip(other.elements).for_each(|(a, b)| *a *= b);
    }
}

impl Vector {
    pub fn new(elements: Vec<f32>) -> Self {
        Self { elements }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn dot(&self, other: &Vector) -> f32 {
        if self.len() != other.len() {
            panic!("Vectors must have the same length for dot product");
        }
        self.elements.iter().zip(&other.elements).map(|(a, b)| a * b).sum()
    }

    pub fn cross(&self, other: &Vector) -> Vector {
        // Write a general cross product for vectors of any size
        if self.len() != other.len() {
            panic!("Vectors must have the same length for cross product");
        }

        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            result.push(self.elements[(i + 1) % self.len()] * other.elements[(i + 2) % self.len()] - self.elements[(i + 2) % self.len()] * other.elements[(i + 1) % self.len()]);
        }
        Vector::new(result)
    }


    pub fn outer_product(&self, other: &Self) -> Matrix {
        let mut matrix = Matrix::new(vec![]);
        for i in 0..self.len() {
            let mut row = vec![];
            for j in 0..other.len() {
                row.push(self.elements[i] * other.elements[j]);
            }
            matrix.rows.push(Vector::new(row));
        }
        matrix
    }

    pub fn magnitude(&self) -> f32 {
        self.elements.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt()
    }

    pub fn normalize(&self) -> Vector {
        let magnitude = self.magnitude();
        Vector::new(self.elements.iter().map(|&x| x / magnitude).collect())
    }

    pub fn distance(&self, other: &Vector) -> f32 {
        self.subtract(other).magnitude()
    }

    pub fn angle(&self, other: &Vector) -> f32 {
        let dot = self.dot(other);
        let magnitude = self.magnitude() * other.magnitude();
        (dot / magnitude).acos()
    }

    pub fn project(&self, other: &Vector) -> Vector {
        let dot = self.dot(other);
        let magnitude_sq = other.magnitude().powi(2);
        other.scalar_multiply(dot / magnitude_sq)
    }

    pub fn reflect(&self, normal: &Vector) -> Vector {
        let dot = self.dot(normal);
        normal.scalar_multiply(2.0 * dot) - self.scalar_multiply(dot)
    }

    pub fn add(&self, other: &Vector) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a + b).collect()
        )
    }

    pub fn add_scalar(&self, scalar: f32) -> Vector {
        Vector::new(self.elements.iter().map(|a| a + scalar).collect())
    }

    pub fn scalar_multiply(&self, scalar: f32) -> Vector {
        Vector::new(self.elements.iter().map(|a| a * scalar).collect())
    }

    pub fn subtract(&self, other: &Vector) -> Vector {
        if self.len() != other.len() {
            panic!("Vectors must have the same length for subtraction");
        }
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| a - b).collect()
        )
    }

    pub fn elementwise_multiply(&self, other: &Vector) -> Vector {
        // round  to 2 decimal places
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(a, b)| (a * b)).collect()
        )
    }

    pub fn elementwise_divide(&self, other: &Vector) -> Vector {
        // Round to 2 decimal places
        let elements = self.elements.iter().zip(&other.elements).map(|(a, b)| (a / b) * 10.0).map(|x| x.round() / 10.0).collect();
        Vector::new(elements)
    }

    pub fn elementwise_sqrt(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| x.sqrt()).collect())
    }

    pub fn sum(&self) -> f32 {
        self.elements.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.len() as f32
    }

    pub fn sigmoid(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
    }

    pub fn sigmoid_derivative(&self) -> Vector {
        self.sigmoid().elementwise_multiply(&self.scalar_multiply(-1.0).add_scalar(1.0))
    }

    pub fn relu(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| x.max(0.0)).collect())
    }

    pub fn relu_derivative(&self) -> Vector {
        Vector::new(self.elements.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect())
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.elements.clone()
    }

    pub fn from_vec(vec: Vec<f32>) -> Vector {
        Self::new(vec)
    }

    pub fn to_string(&self) -> String {
        format!("{:?}", self.elements)
    }

    pub fn from_string(s: &str) -> Result<Vector, std::num::ParseFloatError> {
        let elements: Result<Vec<f32>, _> = s.split(", ").map(str::parse).collect();
        elements.map(Self::new)
    }

    pub fn random(size: usize) -> Vector {
        let mut rng = rand::thread_rng();
        Vector::new((0..size).map(|_| rng.gen_range(-1.0..1.0)).collect())
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Vector {
        Vector::new(self.elements.iter().map(|&x| f(x)).collect())
    }

    pub fn map_with_index(&self, f: impl Fn(f32, usize) -> f32) -> Vector {
        Vector::new(self.elements.iter().enumerate().map(|(i, &x)| f(x, i)).collect())
    }

    pub fn map_with_vector(&self, other: &Vector, f: impl Fn(f32, f32) -> f32) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).map(|(&a, &b)| f(a, b)).collect()
        )
    }

    pub fn map_with_vector_index(&self, other: &Vector, f: impl Fn(f32, f32, usize) -> f32) -> Vector {
        Vector::new(
            self.elements.iter().zip(&other.elements).enumerate().map(|(i, (&a, &b))| f(a, b, i)).collect()
        )
    }

    pub fn unwrap(&self) -> Result<Vector, String> {
        if self.is_finite() {
            Ok(self.clone())
        } else {
            Err("Vector contains non-finite elements".to_string())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn is_finite(&self) -> bool {
        self.elements.iter().all(|&x| x.is_finite())
    }

    pub fn is_zero(&self) -> bool {
        self.elements.iter().all(|&x| x.abs() < f32::EPSILON)
    }

    pub fn is_nan(&self) -> bool {
        self.elements.iter().any(|&x| x.is_nan())
    }

    pub fn is_infinite(&self) -> bool {
        self.elements.iter().any(|&x| x.is_infinite())
    }

    pub fn is_positive(&self) -> bool {
        self.elements.iter().all(|&x| x > 0.0)
    }

    pub fn is_negative(&self) -> bool {
        self.elements.iter().all(|&x| x < 0.0)
    }

    pub fn is_nonpositive(&self) -> bool {
        self.elements.iter().all(|&x| x <= 0.0)
    }

    pub fn is_nonnegative(&self) -> bool {
        self.elements.iter().all(|&x| x >= 0.0)
    }

    pub fn is_sorted(&self) -> bool {
        self.elements.windows(2).all(|w| w[0] <= w[1])
    }

    pub fn set_elements(&mut self, elements: Vec<f32>) {
        self.elements = elements;
    }

    pub fn set_element(&mut self, index: usize, element: f32) {
        self.elements[index] = element;
    }

    pub fn get_element(&self, index: usize) -> f32 {
        self.elements[index]
    }

    pub fn get_elements(&self) -> Vec<f32> {
        self.elements.clone()
    }

    pub fn get_elements_mut(&mut self) -> &mut Vec<f32> {
        &mut self.elements
    }

    pub fn copy(&self) -> Vector {
        Vector::new(self.elements.clone())
    }

    pub fn zeros(size: usize) -> Vector {
        Vector::new(vec![0.0; size])
    }

    pub fn ones(size: usize) -> Vector {
        Vector::new(vec![1.0; size])
    }
}
