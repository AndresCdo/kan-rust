use crate::data_structures::Vector;
use serde::{Deserialize, Serialize};
use std::fs;
use std::process::exit;
use std::slice::{Iter, IterMut};

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

    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.col_count() != other.row_count() {
            print!(
                "Matrix dimensions do not match for multiplication: {}x{} and {}x{}",
                self.row_count(),
                self.col_count(),
                other.row_count(),
                other.col_count()
            );
            return Err("Matrix dimensions do not match for multiplication");
        }
        
        let col_count = other.col_count();
        let mut result = Vec::with_capacity(self.row_count());
        
        for row in &self.rows {
            let mut new_row = Vec::with_capacity(col_count);
            for col in 0..col_count {
                let col_vec: Vec<f32> = other.rows.iter().map(|r| r.elements[col]).collect();
                let dot_product = row.dot(&Vector::new(col_vec));
                new_row.push(dot_product);
            }
            result.push(Vector::new(new_row));
        }
        
        Ok(Matrix::new(result))
    }

    pub fn multiply_with_vector(&self, other: &Vector) -> Result<Vector, &'static str> {
        if self.col_count() != other.len() {
            print!(
                "Matrix and vector dimensions do not match for multiplication: {}x{} and {}",
                self.row_count(),
                self.col_count(),
                other.len()
                );
            return Err("Matrix and vector dimensions do not match for multiplication");
        }

        let mut result = Vec::with_capacity(self.row_count());
        for row in &self.rows {
            let dot_product = row.dot(other);
            result.push(dot_product);
        }
        Ok(Vector::new(result)) 
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            print!(
                "Matrix dimensions do not match for addition: {}x{} and {}x{}",
                self.row_count(),
                self.col_count(),
                other.row_count(),
                other.col_count()
            );
            return Err("Matrix dimensions do not match for addition");
        }

        let rows: Vec<Vector> = self.rows.iter().zip(&other.rows).map(|(a, b)| a.add(b)).collect();
        Ok(Matrix::new(rows))
    }

    pub fn add_scalar(&self, scalar: f32) -> Matrix {
        let rows: Vec<Vector> = self.rows.iter().map(|r| r.add_scalar(scalar)).collect();
        Matrix::new(rows)
    }

    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return Err("Matrix dimensions do not match for subtraction");
        }

        let rows: Vec<Vector> = self.rows.iter().zip(&other.rows).map(|(a, b)| a.subtract(b)).collect();
        Ok(Matrix::new(rows))
    }

    pub fn scalar_multiply(&self, scalar: f32) -> Matrix {
        let rows: Vec<Vector> = self.rows.iter().map(|r| r.scalar_multiply(scalar)).collect();
        Matrix::new(rows)
    }

    pub fn elementwise_multiply(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return Err("Matrix dimensions do not match for element-wise multiplication");
        }

        let rows: Vec<Vector> = self.rows.iter().zip(&other.rows).map(|(a, b)| a.elementwise_multiply(b)).collect();
        Ok(Matrix::new(rows))
    }

    pub fn elementwise_divide(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return Err("Matrix dimensions do not match for element-wise division");
        }

        let rows: Vec<Vector> = self.rows.iter().zip(&other.rows).map(|(a, b)| a.elementwise_divide(b)).collect();
        Ok(Matrix::new(rows))
    }

    pub fn elementwise_sqrt(&self) -> Matrix {
        let rows: Vec<Vector> = self.rows.iter().map(Vector::elementwise_sqrt).collect();
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
        let rows: Vec<Vector> = (0..row_count).map(|_| Vector::random(col_count)).collect();
        Matrix::new(rows)
    }

    pub fn to_vec(&self) -> Vec<Vec<f32>> {
        self.rows.iter().map(Vector::to_vec).collect()
    }

    pub fn from_vec(vec: Vec<Vec<f32>>) -> Matrix {
        Matrix::new(vec.into_iter().map(Vector::new).collect())
    }

    pub fn get_row(&self, index: usize) -> Option<Vector> {
        self.rows.get(index).cloned()
    }

    pub fn get_col(&self, index: usize) -> Result<Vector, &'static str> {
        if index >= self.col_count() {
            return Err("Column index out of bounds");
        }

        let col: Vec<f32> = self.rows.iter().map(|r| r.elements[index]).collect();
        Ok(Vector::new(col))
    }

    pub fn set_row(&mut self, index: usize, row: Vector) -> Result<(), &'static str> {
        if index >= self.row_count() {
            return Err("Row index out of bounds");
        }

        if row.len() != self.col_count() {
            return Err("Row length does not match matrix column count");
        }

        self.rows[index] = row;
        Ok(())
    }

    pub fn set_col(&mut self, index: usize, col: Vector) -> Result<(), &'static str> {
        if index >= self.col_count() {
            return Err("Column index out of bounds");
        }

        if col.len() != self.row_count() {
            return Err("Column length does not match matrix row count");
        }

        for (i, row) in self.rows.iter_mut().enumerate() {
            row.elements[index] = col.elements[i];
        }

        Ok(())
    }

    pub fn set_element(&mut self, row: usize, col: usize, value: f32) -> Result<(), &'static str> {
        if row >= self.row_count() {
            return Err("Row index out of bounds");
        }

        if col >= self.col_count() {
            return Err("Column index out of bounds");
        }

        self.rows[row].elements[col] = value;
        Ok(())
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Matrix {
        let rows: Vec<Vector> = self.rows.iter().map(|r| r.map(&f)).collect();
        Matrix::new(rows)
    }

    pub fn map_with_index(&self, f: impl Fn(f32, usize) -> f32) -> Matrix {
        let rows: Vec<Vector> = self.rows.iter().enumerate().map(|(i, r)| r.map_with_index(|e, j| f(e, j))).collect();
        Matrix::new(rows)
    }

    pub fn map_with_matrix(&self, other: &Matrix, f: impl Fn(f32, f32) -> f32) -> Result<Matrix, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return Err("Matrix dimensions do not match for element-wise mapping");
        }

        let rows: Vec<Vector> = self.rows.iter().zip(&other.rows).map(|(a, b)| a.map_with_vector(b, &f)).collect();
        Ok(Matrix::new(rows))
    }

    pub fn sum(&self) -> f32 {
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

    pub fn mean(&self) -> f32 {
        self.sum() / (self.row_count() * self.col_count()) as f32
    }

    pub fn mean_cols(&self) -> Vector {
        self.sum_cols().scalar_multiply(1.0 / self.row_count() as f32)
    }

    pub fn mean_rows(&self) -> Vector {
        self.sum_rows().scalar_multiply(1.0 / self.col_count() as f32)
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(self).expect("Serialization failed")
    }

    pub fn from_string(s: &str) -> Result<Matrix, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        fs::write(path, self.to_string())
    }

    pub fn load(path: &str) -> Result<Matrix, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        Matrix::from_string(&contents).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn iter(&self) -> Iter<Vector> {
        self.rows.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<Vector> {
        self.rows.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn copy(&self) -> Matrix {
        Matrix::new(self.rows.iter().map(Vector::copy).collect())
    }

    pub fn dot(&self, other: &Vector) -> Result<f32, &'static str> {
        if self.col_count() != other.len() {
            return Err("Matrix and vector dimensions do not match for dot product");
        }

        Ok(self.rows.iter().map(|row| row.dot(other)).sum())
    }

    pub fn dot_with_matrix(&self, other: &Matrix) -> Result<f32, &'static str> {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return Err("Matrix dimensions do not match for dot product");
        }

        Ok(self.rows.iter().zip(&other.rows).map(|(row, other_row)| row.dot(other_row)).sum())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.row_count(), self.col_count())
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[Vector] {
        &self.rows
    }

    pub fn cols(&self) -> Vec<Vector> {
        (0..self.col_count()).map(|i| self.get_col(i).unwrap()).collect()
    }

    pub fn zeros(row_count: usize, col_count: usize) -> Matrix {
        Matrix::new(vec![Vector::zeros(col_count); row_count])
    }

    pub fn ones(row_count: usize, col_count: usize) -> Matrix {
        Matrix::new(vec![Vector::ones(col_count); row_count])
    }
}
