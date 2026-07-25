use kan::network::{
    Kan, KanConfig, KanError, MAX_SERIALIZED_MODEL_BYTES, MAX_TRAINABLE_PARAMETERS,
};

fn training_data() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    (
        vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
        vec![vec![0.3], vec![0.7], vec![1.1]],
    )
}

#[test]
fn trains_a_multilayer_kan_deterministically() {
    let config = KanConfig::new(vec![2, 5, 1], 3).unwrap().with_seed(7);
    let mut model = Kan::try_new(config.clone()).unwrap();
    let mut repeated_model = Kan::try_new(config).unwrap();
    let (inputs, targets) = training_data();

    let initial_loss = model.mse_loss(&inputs, &targets).unwrap();
    model.train(&inputs, &targets, 300, 0.03).unwrap();
    repeated_model.train(&inputs, &targets, 300, 0.03).unwrap();
    let final_loss = model.mse_loss(&inputs, &targets).unwrap();

    assert!(final_loss.is_finite());
    assert!(
        final_loss < initial_loss,
        "expected loss to decrease: initial={initial_loss}, final={final_loss}"
    );
    assert_eq!(model.to_json().unwrap(), repeated_model.to_json().unwrap());
}

#[test]
fn rejects_invalid_model_and_dataset_shapes() {
    assert!(matches!(
        KanConfig::new(vec![2], 3),
        Err(KanError::InvalidShape(_))
    ));
    assert!(matches!(
        KanConfig::new(vec![2, 1], 0),
        Err(KanError::InvalidGridIntervals)
    ));

    let model = Kan::try_new(KanConfig::new(vec![2, 1], 3).unwrap()).unwrap();
    assert!(matches!(
        model.forward(&[0.0]),
        Err(KanError::InputWidthMismatch { .. })
    ));
    assert!(matches!(
        model.mse_loss(&[vec![0.0, 0.0]], &[]),
        Err(KanError::BatchSizeMismatch { .. })
    ));
    assert!(matches!(
        model.mse_loss(&[], &[]),
        Err(KanError::EmptyBatch)
    ));
    assert!(matches!(
        model.mse_loss(&[vec![0.0, 0.0]], &[vec![0.0, 1.0]]),
        Err(KanError::TargetWidthMismatch { .. })
    ));
    assert!(matches!(
        model.forward(&[f32::NAN, 0.0]),
        Err(KanError::NonFiniteInput)
    ));
    assert!(matches!(
        model.mse_loss(&[vec![0.0, 0.0]], &[vec![f32::NAN]]),
        Err(KanError::NonFiniteTarget)
    ));
    assert!(matches!(
        KanConfig::new(vec![2, 1], 3)
            .unwrap()
            .with_domain([1.0, -1.0]),
        Err(KanError::InvalidDomain)
    ));
    assert!(matches!(
        KanConfig::new(vec![1_000, 1_000], 1),
        Err(KanError::ModelTooLarge { .. })
    ));
    assert!(matches!(
        KanConfig::new(vec![usize::MAX, 2], 1),
        Err(KanError::InvalidShape(_))
    ));
    assert!(matches!(
        KanConfig::new(vec![1, 1], usize::MAX),
        Err(KanError::InvalidGridIntervals)
    ));

    let edges_at_limit = MAX_TRAINABLE_PARAMETERS / 8;
    assert!(KanConfig::new(vec![edges_at_limit, 1], 3).is_ok());
    assert!(matches!(
        KanConfig::new(vec![edges_at_limit + 1, 1], 3),
        Err(KanError::ModelTooLarge { .. })
    ));
}

#[test]
fn persists_only_valid_versioned_source_state() {
    let config = KanConfig::new(vec![2, 3, 1], 4)
        .unwrap()
        .with_domain([-2.0, 2.0])
        .unwrap()
        .with_seed(11);
    let model = Kan::try_new(config).unwrap();
    let input = vec![0.2, -0.7];
    let expected = model.forward(&input).unwrap();

    let encoded = model.to_json().unwrap();
    let restored = Kan::from_json(&encoded).unwrap();

    assert_eq!(restored.shape(), &[2, 3, 1]);
    assert_eq!(restored.forward(&input).unwrap(), expected);
    assert!(matches!(
        Kan::from_json(r#"{"layers":[]}"#),
        Err(KanError::LegacyUnversionedModel)
    ));
    assert!(matches!(
        Kan::from_json(r#"{"format_version":1,"model":"#),
        Err(KanError::Serialization(_))
    ));

    let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    value["format_version"] = serde_json::json!(99);
    assert!(matches!(
        Kan::from_json(&value.to_string()),
        Err(KanError::UnsupportedFormatVersion(99))
    ));

    let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    value["unexpected"] = serde_json::json!(true);
    assert!(matches!(
        Kan::from_json(&value.to_string()),
        Err(KanError::Serialization(_))
    ));

    let duplicate_format = encoded.replacen(
        r#""format":"kan-rust","#,
        r#""format":"kan-rust","format":"kan-rust","#,
        1,
    );
    assert!(matches!(
        Kan::from_json(&duplicate_format),
        Err(KanError::Serialization(_))
    ));
    let duplicate_shape = encoded.replacen(
        r#""shape":[2,3,1],"#,
        r#""shape":[2,3,1],"shape":[2,3,1],"#,
        1,
    );
    assert!(matches!(
        Kan::from_json(&duplicate_shape),
        Err(KanError::Serialization(_))
    ));

    let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    value["model"]["layers"][0]["base_weights"][0] = serde_json::json!([0.0]);
    assert!(matches!(
        Kan::from_json(&value.to_string()),
        Err(KanError::MalformedModel(_))
    ));

    let mut value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    value["model"]["layers"][0]["coefficients"][0][0]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(matches!(
        Kan::from_json(&value.to_string()),
        Err(KanError::MalformedModel(_))
    ));
    assert!(matches!(
        Kan::from_json(&" ".repeat(MAX_SERIALIZED_MODEL_BYTES + 1)),
        Err(KanError::SerializedModelTooLarge { .. })
    ));
}

#[test]
fn initialization_is_reproducible_and_parameter_count_is_exact() {
    let config = KanConfig::new(vec![2, 3, 1], 4).unwrap().with_seed(42);
    let first = Kan::try_new(config.clone()).unwrap();
    let second = Kan::try_new(config).unwrap();
    let different = Kan::try_new(KanConfig::new(vec![2, 3, 1], 4).unwrap().with_seed(43)).unwrap();

    assert_eq!(first.to_json().unwrap(), second.to_json().unwrap());
    assert_ne!(first.to_json().unwrap(), different.to_json().unwrap());
    assert_eq!(first.parameter_count(), 81);
}

#[test]
fn seeded_initialization_matches_the_stable_parameter_fixture() {
    let model = Kan::try_new(KanConfig::new(vec![1, 1], 1).unwrap().with_seed(42)).unwrap();
    let value: serde_json::Value = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    let layer = &value["model"]["layers"][0];

    assert_eq!(
        layer["base_weights"][0][0].as_f64().unwrap() as f32,
        0.483_129_86
    );
    assert_eq!(layer["spline_weights"][0][0].as_f64().unwrap() as f32, 1.0);
    let coefficients: Vec<f32> = layer["coefficients"][0][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(
        coefficients,
        vec![
            -0.006_801_792_4,
            -0.004_427_977,
            -0.003_116_186_2,
            -0.009_239_397
        ]
    );
}

#[test]
fn failed_numerical_update_is_atomic() {
    let mut model = Kan::try_new(KanConfig::new(vec![1, 1], 3).unwrap()).unwrap();
    let before = model.to_json().unwrap();

    assert!(matches!(
        model.train_step(&[vec![1.0]], &[vec![f32::MAX]], f32::MAX),
        Err(KanError::NumericalFailure(_))
    ));
    assert_eq!(model.to_json().unwrap(), before);

    assert!(matches!(
        model.train(&[vec![1.0]], &[vec![0.0]], 0, 0.0),
        Err(KanError::InvalidLearningRate)
    ));
}
