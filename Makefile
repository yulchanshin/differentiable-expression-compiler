MANIFEST := --manifest-path engine/Cargo.toml

.PHONY: build test run bench

build:
	cargo build $(MANIFEST)

test:
	cargo test $(MANIFEST)

run:
	cargo run $(MANIFEST)

bench:
	cargo bench $(MANIFEST)
