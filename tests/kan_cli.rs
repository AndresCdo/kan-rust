use kan::network::MAX_SERIALIZED_MODEL_BYTES;
use std::fs;
use std::process::Command;

#[test]
fn demo_writes_a_model_that_inspect_can_validate() {
    let model_path = std::env::temp_dir().join(format!("kan-rust-{}.json", std::process::id()));
    let binary = env!("CARGO_BIN_EXE_kan");

    let demo = Command::new(binary)
        .arg(&model_path)
        .output()
        .expect("KAN demo process should start");
    assert!(
        demo.status.success(),
        "demo stderr: {}",
        String::from_utf8_lossy(&demo.stderr)
    );
    assert!(model_path.is_file());

    let inspect = Command::new(binary)
        .arg("inspect")
        .arg(&model_path)
        .output()
        .expect("KAN inspect process should start");
    assert!(
        inspect.status.success(),
        "inspect stderr: {}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    assert!(String::from_utf8_lossy(&inspect.stdout).contains("shape=[2, 5, 1]"));

    fs::remove_file(model_path).expect("temporary model should be removable");
}

#[test]
fn inspect_rejects_an_oversized_model_file() {
    let model_path =
        std::env::temp_dir().join(format!("kan-rust-oversized-{}.json", std::process::id()));
    let file = fs::File::create(&model_path).expect("temporary model should be creatable");
    let oversized_length = u64::try_from(MAX_SERIALIZED_MODEL_BYTES)
        .unwrap()
        .checked_add(1)
        .unwrap();
    file.set_len(oversized_length)
        .expect("temporary model should be sizeable");

    let inspect = Command::new(env!("CARGO_BIN_EXE_kan"))
        .arg("inspect")
        .arg(&model_path)
        .output()
        .expect("KAN inspect process should start");
    assert!(!inspect.status.success());
    assert!(String::from_utf8_lossy(&inspect.stderr).contains("input limit"));

    fs::remove_file(model_path).expect("temporary model should be removable");
}
