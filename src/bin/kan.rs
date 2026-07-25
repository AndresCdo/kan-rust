use kan::network::{Kan, KanConfig, MAX_SERIALIZED_MODEL_BYTES};
use std::error::Error;
use std::fs;
use std::io::Read;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments
        .first()
        .is_some_and(|argument| argument == "inspect")
    {
        let path = arguments.get(1).ok_or("usage: kan inspect <model.json>")?;
        let model = Kan::from_json(&read_model(path)?)?;
        println!(
            "shape={:?}, parameters={}",
            model.shape(),
            model.parameter_count()
        );
        return Ok(());
    }

    let output_path = arguments
        .first()
        .map(String::as_str)
        .unwrap_or("model.json");
    let mut model = Kan::try_new(KanConfig::new(vec![2, 5, 1], 3)?.with_seed(7))?;
    let inputs = vec![
        vec![-0.8, -0.6],
        vec![-0.4, 0.3],
        vec![0.1, -0.2],
        vec![0.5, 0.4],
        vec![0.8, -0.1],
    ];
    let targets = vec![vec![-1.4], vec![-0.1], vec![-0.1], vec![0.9], vec![0.7]];
    let initial_loss = model.mse_loss(&inputs, &targets)?;
    model.train(&inputs, &targets, 300, 0.03)?;
    let final_loss = model.mse_loss(&inputs, &targets)?;

    fs::write(output_path, model.to_json()?)?;
    println!("loss: {initial_loss:.6} -> {final_loss:.6}");
    println!("saved validated KAN model to {output_path}");
    Ok(())
}

fn read_model(path: &str) -> Result<String, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err("model input must be a regular file".into());
    }
    let maximum_bytes = u64::try_from(MAX_SERIALIZED_MODEL_BYTES)?;
    if metadata.len() > maximum_bytes {
        return Err(format!(
            "model exceeds the {} byte input limit",
            MAX_SERIALIZED_MODEL_BYTES
        )
        .into());
    }

    read_bounded_model(file, MAX_SERIALIZED_MODEL_BYTES)
}

fn read_bounded_model<R: Read>(reader: R, maximum_bytes: usize) -> Result<String, Box<dyn Error>> {
    let sentinel_bytes = maximum_bytes
        .checked_add(1)
        .ok_or("model input limit cannot be represented")?;
    let read_limit = u64::try_from(sentinel_bytes)?;
    let mut encoded = String::new();
    reader.take(read_limit).read_to_string(&mut encoded)?;
    if encoded.len() > maximum_bytes {
        return Err(format!("model exceeds the {} byte input limit", maximum_bytes).into());
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::read_bounded_model;
    use std::io::Cursor;

    #[test]
    fn bounded_reader_accepts_the_limit_and_rejects_one_byte_more() {
        assert_eq!(read_bounded_model(Cursor::new(b"1234"), 4).unwrap(), "1234");
        assert!(read_bounded_model(Cursor::new(b"12345"), 4).is_err());
    }
}
