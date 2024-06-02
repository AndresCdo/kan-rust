use crate::data_structures::Vector;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: Vec<Vector>,
}

impl Matrix {
    pub fn new(rows: Vec<Vector>) -> Self {
        Matrix { rows }
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn col_count(&self) -> usize {
        self.rows.first().map_or(0, |row| row.len())
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        let col_count = other.col_count();
        let mut result = Vec::with_capacity(self.row_count());
        for row in &self.rows {
            let mut new_row = Vec::with_capacity(col_count);
            for col in 0..col_count {
                let col_vec: Vec<f64> = other.rows.iter().map(|r| r.elements[col]).collect();
                let dot_product = row.dot(&Vector::new(col_vec));
                new_row.push(dot_product);
            }
            result.push(Vector::new(new_row));
        }
        Matrix::new(result)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.iter().zip(&other.rows).map(|(a, b)| a.add(b)).collect();
        Matrix::new(rows)
    }

    pub fn add_scalar(&self, scalar: f64) -> Matrix {
        let rows = self.rows.iter().map(|r| r.add_scalar(scalar)).collect();
        Matrix::new(rows)
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.iter().zip(&other.rows).map(|(a, b)| a.subtract(b)).collect();
        Matrix::new(rows)
    }

    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        let rows = self.rows.iter().map(|r| r.scalar_multiply(scalar)).collect();
        Matrix::new(rows)
    }

    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.iter().zip(&other.rows).map(|(a, b)| a.elementwise_multiply(b)).collect();
        Matrix::new(rows)
    }

    pub fn elementwise_divide(&self, other: &Matrix) -> Matrix {
        let rows = self.rows.iter().zip(&other.rows).map(|(a, b)| a.elementwise_divide(b)).collect();
        Matrix::new(rows)
    }

    pub fn elementwise_sqrt(&self) -> Matrix {
        let rows = self.rows.iter().map(Vector::elementwise_sqrt).collect();
        Matrix::new(rows)
    }

    pub fn transpose(&self) -> Matrix {
        let row_count = self.row_count();
        let col_count = self.col_count();
        let mut transposed = vec![vec![0.0; row_count]; col_count];
        for i in 0..row_count {
            for j in 0..col_count {
                transposed[j][i] = self.rows[i].elements[j];
            }
        }
        Matrix::new(transposed.into_iter().map(Vector::new).collect())
    }

    pub fn random(row_count: usize, col_count: usize) -> Matrix {
        // Non NULL random number generator
        rand::thread_rng();
        let mut rows = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            let row = Vector::random(col_count);
            rows.push(row);
        }
        Matrix::new(rows)
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        self.rows.iter().map(|r| r.to_vec()).collect()
    }

    pub fn from_vec(vec: Vec<Vec<f64>>) -> Matrix {
        Matrix::new(vec.into_iter().map(Vector::new).collect())
    }

    pub fn get_row(&self, index: usize) -> Vector {
        self.rows[index].clone()
    }

    pub fn get_col(&self, index: usize) -> Vector {
        Vector::new(self.rows.iter().map(|r| r.elements[index]).collect())
    }

    pub fn set_row(&mut self, index: usize, row: Vector) {
        self.rows[index] = row;
    }

    pub fn set_col(&mut self, index: usize, col: Vector) {
      for (i, row) in self.rows.iter_mut().enumerate() {
        row.elements[index] = col.elements[i];
      }
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Matrix {
      let rows = self.rows.iter().map(|r| r.map(&f)).collect();
      Matrix::new(rows)
    }

    pub fn map_with_index(&self, f: impl Fn(f64, usize) -> f64) -> Matrix {
      let rows = self.rows.iter().enumerate().map(|(_i, r)| r.map_with_index(|e, j| f(e, j))).collect();
      Matrix::new(rows)
    }

    pub fn map_with_matrix(&self, other: &Matrix, f: impl Fn(f64, f64) -> f64) -> Matrix {
        let rows = self.rows.iter().zip(&other.rows).map(|(a, b)| a.map_with_vector(b, &f)).collect();
        Matrix::new(rows)
    }

    pub fn sum(&self) -> f64 {
        self.rows.iter().map(Vector::sum).sum()
    }

    pub fn sum_cols(&self) -> Vector {
        let col_count = self.col_count();
        let mut result = vec![0.0; col_count];
        for row in &self.rows {
            for (i, &e) in row.elements.iter().enumerate() {
                result[i] += e;
            }
        }
        Vector::new(result)
    }

    pub fn sum_rows(&self) -> Vector {
        Vector::new(self.rows.iter().map(Vector::sum).collect())
    }

    pub fn mean(&self) -> f64 {
        self.sum() / (self.row_count() * self.col_count()) as f64
    }

    pub fn mean_cols(&self) -> Vector {
        self.sum_cols().scalar_multiply(1.0 / self.row_count() as f64)
    }

    pub fn mean_rows(&self) -> Vector {
        self.sum_rows().scalar_multiply(1.0 / self.col_count() as f64)
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn from_string(s: &str) -> Result<Matrix, std::num::ParseFloatError> {
        let rows: Result<Vec<Vector>, _> = s.split("\n").map(Vector::from_string).collect();
        rows.map(Matrix::new)
    }

    pub fn from_str(s: &str) -> Matrix {
        serde_json::from_str(s).unwrap()
    }

    pub fn save(&self, path: &str) {
        std::fs::write(path, self.to_string()).unwrap();
    }

    pub fn load(path: &str) -> Matrix {
        let contents = std::fs::read_to_string(path).unwrap();
        Matrix::from_str(&contents)
    }
}
