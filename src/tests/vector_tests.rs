use crate::data_structures::Vector;

#[test]
fn test_vector_operations() {
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

    assert_eq!(v1.add(&v2), Vector::new(vec![5.0, 7.0, 9.0]));
    assert_eq!(v1.subtract(&v2), Vector::new(vec![-3.0, -3.0, -3.0]));
    assert_eq!(v1.dot(&v2), 32.0);
}
