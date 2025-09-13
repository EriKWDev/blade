set -e

cargo lcheck
cargo lcheck --target aarch64-apple-darwin
RUSTFLAGS="--cfg gles" CARGO_TARGET_DIR=./target-gl cargo lcheck
