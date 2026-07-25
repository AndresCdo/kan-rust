# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-25

### Added

- Add a validated, deterministic fixed-grid KAN API.
- Add complete multilayer gradients for spline coefficients and edge weights.
- Add versioned JSON persistence, resource limits, and a KAN CLI demo.
- Add mathematical, gradient, training, persistence, and CLI regression tests.

### Changed

- Use canonical cubic B-splines with exterior-extended uniform knots.
- Make invalid shapes, datasets, numerical updates, and model files fallible.
- Track `Cargo.lock`, update dependencies, and enforce all quality gates in CI.

### Security

- Bound model parameters, JSON size and complexity, and CLI model-file reads.

### Removed

- Remove the unvalidated `KanNetwork` and `create_kan` public API.
- Remove unsupported regularization, grid extension, pruning, and symbolic APIs.

Unversioned 0.1 KAN model data is intentionally unsupported because its spline
and persistence semantics cannot be migrated reliably.

[0.2.0]: https://github.com/AndresCdo/kan-rust/releases/tag/v0.2.0
