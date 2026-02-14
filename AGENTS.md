# AGENTS.md - Documentación para Agentes IA

Este archivo contiene instrucciones para agentes IA que trabajan en este proyecto.

## Información del Proyecto

- **Nombre**: KAN (Kolmogorov-Arnold Networks) Rust Implementation
- **Lenguaje**: Rust
- **Tipo**: Librería + Binario
- **Repositorio**: https://github.com/AndresCdo/kan-rust

## Estructura del Proyecto

```
kan-rust/
├── src/
│   ├── bin/kan.rs           # Punto de entrada binario
│   ├── data_structures/     # Estructuras de datos core
│   │   ├── vector.rs        # Operaciones vectoriales
│   │   ├── matrix.rs       # Operaciones matriciales
│   │   ├── layer.rs        # Capa MLP (legacy)
│   │   └── spline.rs       # Implementación B-spline
│   ├── network/             # Implementaciones de redes
│   │   ├── network.rs      # Red MLP (legacy)
│   │   └── kan.rs          # Implementación KAN ✓
│   ├── utils/              # Utilidades
│   └── tests/              # Tests unitarios
├── docs/
│   └── TECHNICAL.md        # Documentación técnica
├── Cargo.toml
└── README.md
```

## Comandos de Desarrollo

### Compilación

```bash
# Desarrollo
cargo build

# Release
cargo build --release

# Verificar sin compilar
cargo check
```

### Testing

```bash
# Todos los tests
cargo test

# Tests de librería
cargo test --lib

# Tests con output
cargo test -- --nocapture

# Coverage
cargo tarpaulin
```

### Linting

```bash
# Fix automático
cargo fix

# Clippy
cargo clippy -- -D warnings
```

### Documentación

```bash
# Generar docs
cargo doc --open

# Docs sin abrir
cargo doc
```

## Arquitectura KAN

El proyecto implementa **Kolmogorov-Arnold Networks** según el paper:

- **Paper**: [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v5)
- **Repo original Python**: https://github.com/KindXiaoming/pykan

### Características Implementadas

| Feature | Estado | Archivo |
|---------|--------|---------|
| B-spline activation | ✅ | `spline.rs` |
| SiLU base function | ✅ | `spline.rs` |
| Forward propagation | ✅ | `kan.rs` |
| Backpropagation | ✅ | `kan.rs` |
| Grid extension | ✅ | `kan.rs` |
| L1 regularization | ✅ | `kan.rs` |
| Entropy regularization | ✅ | `kan.rs` |
| Node pruning | ✅ | `kan.rs` |
| Symbolic fixing | 🔄 | placeholder |

## API Principal

### Crear KAN

```rust
use kan::network::kan::create_kan;

let kan = create_kan(&[2, 5, 1], 3); // shape, grid_size
```

### Forward Pass

```rust
let input = vec![0.5, 0.3];
let output = kan.forward(&input);
```

### Entrenamiento

```rust
kan.train(&inputs, &targets, epochs, learning_rate);
```

### Pérdidas

```rust
let mse = kan.mse_loss(&inputs, &targets);
let total = kan.total_loss(&inputs, &targets); // con regularización
```

## Estándares de Código

- Rust 2021 edition
- Warnings como errores en CI
- Tests unitarios para nuevas features
- Documentación para funciones públicas
- Nombres en snake_case
- Types concretos preferidos a inferencia

## Testing Requirements

Antes de commit:

1. `cargo test` debe pasar
2. `cargo clippy` sin warnings
3. `cargo fmt` formateado
4. Documentación actualizada si hay cambios API

## Dependencias

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"
indicatif = "0.15"
ctrlc = "3.1"
```

## Notas para Agentes

- El código legacy (MLP) está en `layer.rs` y `network.rs`
- La nueva implementación KAN está en `spline.rs` y `kan.rs`
- Los tests están en los módulos mismos (inline con `#[cfg(test)]`)
- Hay warnings de lifetime en `matrix.rs` que necesitan fix
- La implementación de backpropagation en KAN es parcial (solo primera capa)
