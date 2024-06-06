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

    let result = m1.multiply(&m2).unwrap();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![19.0, 22.0]),
        Vector::new(vec![43.0, 50.0])
    ]));

    let v1 = Vector::new(vec![1.0, 2.0]);
    let v2 = Vector::new(vec![3.0, 4.0]);

    let result = m1.multiply_with_vector(&v1).unwrap();
    assert_eq!(result,  Vector::new(vec![5.0, 11.0]));
    let result = m1.multiply_with_vector(&v2).unwrap();
    assert_eq!(result,  Vector::new(vec! [11.0, 25.0]));

    let result = m2.multiply_with_vector(&v1).unwrap();
    assert_eq!(result,  Vector::new(vec![17.0, 23.0]));
    let result = m2.multiply_with_vector(&v2).unwrap();
    assert_eq!(result,  Vector::new(vec![39.0, 53.0]));

    let result = m1.add(&m2).unwrap();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![6.0, 8.0]),
        Vector::new(vec![10.0, 12.0])
    ]));

    let result = m1.subtract(&m2).unwrap();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![-4.0, -4.0]),
        Vector::new(vec![-4.0, -4.0])
    ]));

    let result = m1.transpose();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![1.0, 3.0]),
        Vector::new(vec![2.0, 4.0])
    ]));

    let result = m1.elementwise_multiply(&m2).unwrap();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![5.0, 12.0]),
        Vector::new(vec![21.0, 32.0])
    ]));

    let result = m1.elementwise_divide(&m2).unwrap();
    assert_eq!(result, Matrix::new(vec![
        Vector::new(vec![0.2, 0.3]),
        Vector::new(vec![0.4, 0.5])
    ]));
}
