use crate::data_structures::{Matrix, Vector};

#[test]
fn test_matrix_operations() {
    let m1 = Matrix::new(vec![
        Vector::new(vec![1.0, 2.0]),
        Vector::new(vec![3.0, 4.0])
    ]);

    let m2 = Matrix::new(vec![
        Vector::new(vec![5.0, 6.0]),
        Vector::new(vec![7.0, 8.0])
    ]);

    let result = m1.multiply(&m2);
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![19.0, 22.0]),
        Vector::new(vec![43.0, 50.0])
    ]));
}
