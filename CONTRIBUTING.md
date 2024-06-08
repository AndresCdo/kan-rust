# Contributing to Kolmogorov-Arnold Network (KAN) in Rust

Thank you for considering contributing to our Kolmogorov-Arnold Network (KAN) project! We welcome contributions from the community to improve and extend this project. This document provides guidelines to ensure that contributions are made smoothly and effectively.

## How Can I Contribute?

### Reporting Bugs

If you encounter a bug, please report it by opening an issue. Ensure you include:
- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected and actual behavior.
- Any relevant logs, screenshots, or code snippets.

### Suggesting Enhancements

If you have ideas for enhancements or new features, please open an issue. Include:
- A clear and descriptive title.
- A detailed description of the enhancement or feature.
- Any relevant context, including why this enhancement would be useful.

### Submitting Pull Requests

We welcome pull requests (PRs)! To submit a PR:

1. **Fork the Repository**: Create a fork of the repository on GitHub.
2. **Clone the Fork**: Clone your forked repository to your local machine.
    ```sh
    git clone https://github.com/AndresCdo/kan-rust.git
    ```
3. **Create a Branch**: Create a new branch for your changes.
    ```sh
    git checkout -b feature/your-feature-name
    ```
4. **Make Changes**: Implement your changes, ensuring they adhere to the project's coding standards.
5. **Commit Changes**: Commit your changes with a clear and concise commit message.
    ```sh
    git commit -m "Add feature: description of the feature"
    ```
6. **Push Changes**: Push your changes to your fork on GitHub.
    ```sh
    git push origin feature/your-feature-name
    ```
7. **Open a Pull Request**: Go to the original repository and open a pull request from your branch. Include a detailed description of your changes and reference any related issues.

### Code Style

Please ensure that your code adheres to the following style guidelines:
- **Rustfmt**: Use `rustfmt` to format your code. This can be done by running:
    ```sh
    cargo fmt
    ```
- **Clippy**: Use `clippy` to catch common mistakes and improve code quality. Run:
    ```sh
    cargo clippy
    ```
- **Documentation**: Add comments and documentation to your code where appropriate, especially for public APIs.

### Testing

Ensure that your changes are well-tested:
- Write unit tests for new features or bug fixes.
- Run all tests to ensure they pass:
    ```sh
    cargo test
    ```

## Code of Conduct

Please note that this project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Help

If you have any questions or need help, feel free to open an issue or reach out to the maintainers.

Thank you for contributing to the Kolmogorov-Arnold Network (KAN) project in Rust! Your contributions are greatly appreciated.
