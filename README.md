# Gradient Engine

A differentiable expression compiler and durable optimization service, written in Rust.

Gradient Engine takes a math function over several variables (for example `f(x, y) = sin(x*y) + x^2`), compiles it into a computational graph (a DAG), and can then evaluate it, differentiate it *exactly* via reverse-mode automatic differentiation (full gradient and Jacobian), and drive iterative solvers (Newton's method, inverse kinematics) on top of it.

The bulk of the code is a real compiler pipeline: lexer, parser, graph IR, optimization passes, execution, plus (in later tiers) a durable-orchestration layer and a full-stack visualizer. The calculus is the payload; the substance is PL/compilers, numerics, and distributed systems. It is deliberately aimed at solvers and a robot arm rather than neural nets.

> **Status:** early. The Rust autodiff core (graph arena, forward/backward passes, Jacobian, finite-difference validation) is implemented and tested. The parser, optimizer, symbolic-diff section, solvers, HTTP service, and the Go/TypeScript tiers are planned. See [Roadmap and status](#roadmap-and-status).

The single mental model to hold onto:

> A computational graph is a DAG. The forward pass is a topological-order evaluation. The backward pass is a reverse-topological-order traversal that accumulates adjoints.

Everything hangs off that spine.

## Demo

_Coming soon._ A CLI demo and an animated browser visualizer (forward values, then backward adjoints, animated on the actual graph, plus an inverse-kinematics arm reaching for a clicked target) are planned once the parser and service layers land. Until then, the test suite exercises the engine end to end.

## Setup

Requirements: a recent Rust toolchain (edition 2024; install via [rustup](https://rustup.rs)).

```sh
git clone <this-repo>
cd grad_engine

make build    # cargo build  --manifest-path engine/Cargo.toml
make test     # cargo test   --manifest-path engine/Cargo.toml   (runs the finite-difference oracle)
make run      # cargo run    --manifest-path engine/Cargo.toml   (CLI entry, placeholder for now)
make bench    # cargo bench  --manifest-path engine/Cargo.toml   (planned)
```

The only external dependency today is `rand` (used by the validation harness). `axum`, `tokio`, and `serde` arrive with the service ticket.

## Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │                 BROWSER (SPA)                  │  [Tier 2/3]
                    │  TypeScript + React + Vite                     │
                    │  Graph Visualizer (D3+dagre) · IK Arm · Plot   │
                    └───────────────┬───────────────────────────────┘
                                    │  WebSocket (JSON frames) + REST
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │                  GO SERVICE LAYER (BFF)  [Tier 2]          │
        │  REST · WebSocket · Temporal client · Postgres            │
        └───────┬──────────────────────┬───────────────────┬────────┘
                │ HTTP/JSON             │ Temporal          │ SQL
                ▼                       ▼                   ▼
   ┌────────────────────────┐  ┌──────────────────┐  ┌──────────────┐
   │  RUST ENGINE (service) │  │  TEMPORAL SERVER │  │  POSTGRES    │
   │  lexer > parser >      │  │  durable solver  │  │  functions,  │
   │  graph IR > optimizer  │  │  workflows       │  │  runs        │
   │  > forward/reverse AD  │  └──────────────────┘  └──────────────┘
   │  > LU/Newton/IK        │
   └────────────────────────┘
        ▲
        │  Tier 1 stops here: the Rust engine alone (via CLI or a thin
        │  HTTP endpoint) is already a complete, standalone project.
```

The engine is a self-contained Rust project. Go, Temporal, and the browser visualizer are additive tiers that turn a great *engine* into a great *system*. The Rust/Go boundary sits at a genuine compute-versus-serving seam (the Rust engine is its own service the Go layer calls), not an artificial split through the middle of one component.

The project is sequenced math-first, parser-last: the autodiff core is built on hand-wired graphs before the parser exists, so the hardest Rust lesson (the index-based arena graph) is learned on the most interesting part.

## File structure

```
grad_engine/
├── README.md                   # this file
├── roadmap.md                  # the full product roadmap, phases, and per-ticket detail
├── Makefile                    # build / test / run / bench shortcuts
│
├── engine/                     # RUST: the whole compiler + AD + solvers
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs             # CLI entry (Tier 1) / server entry (Tier 2)
│   │   ├── lib.rs              # crate root, module re-exports
│   │   ├── graph/
│   │   │   ├── node.rs         # Node struct + OpType enum (the data model)
│   │   │   ├── arena.rs        # Vec<Node> arena + index-based construction   [done]
│   │   │   └── topo.rs         # topological-ordering utilities
│   │   ├── autodiff/
│   │   │   ├── forward.rs      # forward evaluation                            [done]
│   │   │   ├── backward.rs     # reverse-mode backward pass (the AD core)      [done]
│   │   │   ├── jacobian.rs     # multi-output Jacobian assembly                [done]
│   │   │   └── trace.rs        # emit the frontend trace contract (serde)      [planned]
│   │   ├── ops/derivatives.rs  # per-op forward + local-derivative table
│   │   ├── optimize/           # constfold / cse / deadcode                    [planned]
│   │   ├── parse/              # lexer / ast / parser / lower (hash-consing)   [planned]
│   │   ├── linalg/lu.rs        # LU decomposition                              [planned]
│   │   ├── solver/             # newton.rs, ik.rs                              [planned]
│   │   └── error.rs            # EngineError enum                              [done]
│   └── tests/
│       └── finite_diff.rs      # the correctness oracle                        [done]
│
├── server/                     # GO: service layer / BFF          [Tier 2, planned]
├── worker/                     # GO: Temporal workflows           [Tier 2, planned]
├── web/                        # TypeScript + React + Vite        [Tier 2/3, planned]
└── bench/                      # reverse-vs-forward cost, node-count reduction [planned]
```

## The math and the code

### Automatic differentiation

Automatic differentiation is a third thing, distinct from two lookalikes:

- **Symbolic differentiation** manipulates formulas into formulas. Exact, but the expressions blow up.
- **Numerical differentiation** (finite differences) approximates `f'(x) ≈ (f(x+h) − f(x)) / h`: one extra evaluation per input, and only approximate.
- **Automatic differentiation** applies the chain rule *locally* at each primitive op over the graph. Exact, at roughly one function-evaluation's cost.

This engine implements **reverse mode** (the many-inputs, few-outputs case, the same idea as backprop). A planned section also implements symbolic differentiation, precisely so the "expressions blow up" claim can be measured rather than asserted.

### The graph (arena)

The computation DAG is a single `Vec<Node>` arena. Nodes reference each other by `usize` **index**, not by pointer or `Rc`:

```rust
struct Node { op: OpType, inputs: Vec<usize>, value: f64, adjoint: f64 }
struct Graph { nodes: Vec<Node> }
```

Builder helpers (`var`, `constant`, `add`, `sub`, `mul`, `div`, `neg`, `pow`, `sin`, `cos`, `exp`, `ln`) each push a node and return its index, so `f(x, y) = sin(x*y) + x^2` is built by hand in a few readable lines:

```rust
let x  = g.var("x".into());
let y  = g.var("y".into());
let xy = g.mul(x, y);
let s  = g.sin(xy);
let x2 = g.pow(x, 2.0);
let f  = g.add(s, x2);       // f(x, y) = sin(x*y) + x^2
```

A node used twice (a shared subexpression) is simply one index that appears in two `inputs` lists: no ownership conflict, computed once.

### Forward and backward passes

- **Forward** (`Graph::forward(&inputs) -> Result<f64, EngineError>`): fill each node's `value` in topological order and return the output value. Because the builders always push a node's inputs before the node itself, plain index order is already a valid topological order (forward), and reverse index order serves the backward pass.

- **Backward** (`Graph::backward(output) -> Result<HashMap<String, f64>, EngineError>`): the reverse-mode core, which *is* the multivariable chain rule over the DAG. Seed the output's adjoint to 1 (`∂f/∂f = 1`), then walk nodes in reverse topological order; each node pushes its adjoint into its inputs' adjoint slots via that op's local derivative, **accumulating with `+=`** (the sum-over-paths rule, which is why shared nodes accumulate rather than assign). After one pass, each variable node's adjoint holds `∂f/∂that_var`, returned as a gradient map.

The local-derivative contract (incoming adjoint `ḡ`):

| Op | Forward | Adjoint to inputs |
|---|---|---|
| `add(a,b)` | `a+b` | `ā += ḡ`; `b̄ += ḡ` |
| `sub(a,b)` | `a−b` | `ā += ḡ`; `b̄ += −ḡ` |
| `mul(a,b)` | `a*b` | `ā += ḡ*b`; `b̄ += ḡ*a` |
| `div(a,b)` | `a/b` | `ā += ḡ/b`; `b̄ += −ḡ*a/(b*b)` |
| `pow(a,k)` | `a^k` | `ā += ḡ*k*a^(k−1)` |
| `sin(a)` | `sin a` | `ā += ḡ*cos a` |
| `cos(a)` | `cos a` | `ā += −ḡ*sin a` |
| `exp(a)` | `exp a` | `ā += ḡ*exp a` |
| `ln(a)` | `ln a` | `ā += ḡ/a` |
| `neg(a)` | `−a` | `ā += −ḡ` |

### Jacobian

For `f: ℝⁿ → ℝᵐ`, the Jacobian `J[i][j] = ∂fᵢ/∂xⱼ`. Reverse mode produces **one row per backward pass** (seed output `i`'s adjoint to 1, the rest 0). `Graph::jacobian(outputs, vars) -> Result<Vec<Vec<f64>>, EngineError>` shares one forward pass across all rows and runs one backward pass per output. The solvers consume this matrix.

### Errors

Fallible passes return `EngineError` for expected, input-dependent failures: `UnknownVariable`, `DivByZero`, `DomainError` (for example `ln(x ≤ 0)`). Programmer bugs (empty graph, bad node index) panic instead; the two categories are kept deliberately separate.

## Major design decisions

- **Arena of indices, not `Rc<RefCell<Node>>`.** A multi-parent graph is the classic case where beginners reach for reference-counted, interior-mutable nodes and inherit runtime borrow panics. Instead, every node lives in one owner (the `Vec`), indices are `Copy` handles passed around freely, and single ownership is enforced at compile time. This is the key Rust lesson of the project.

- **Math-first, parser-last.** Autodiff is built and validated on hand-constructed graphs before any lexer or parser exists, so the graph and the calculus are solid before syntax is layered on top.

- **Finite differences as the correctness oracle.** AD bugs are subtle: a wrong sign or a missing accumulation on a shared node produces wrong numbers without crashing. Every gradient is auto-checked against central finite differences (`< 1e-5`); this harness is green before anything is built on top of it.

- **Rust owns compute, Go owns orchestration.** The language boundary is a real compute-versus-serving seam. Rust holds the exact/fast numerics and the compiler; Go holds concurrent service glue and Temporal (which is Go-native).

- **A first-class trace contract.** The visualizer animates whatever the engine emits, so the trace (two ordered arrays, forward values in topological order and backward adjoints in reverse, over a shared node list) is designed as an engine output, not retrofitted.

## Features

Implemented today:

- Index-based arena graph with ergonomic hand-construction builders.
- Forward evaluation in topological order.
- Reverse-mode automatic differentiation (exact gradients in one backward pass).
- Multi-output Jacobian assembly.
- Recoverable error handling for unknown variables, division by zero, and domain violations.
- Finite-difference validation harness.

Planned (see the roadmap): symbolic differentiation and higher-order derivatives, the lexer/Pratt-parser/lowering front end, optimization passes (constant folding, CSE, dead-code elimination), LU/Newton/IK solvers, a JSON trace emitter, an HTTP service, and the Go + Temporal + TypeScript tiers.

## HTTP API

_Planned (Tier 1 service ticket)._ The engine will be wrapped in a small `axum` + `tokio` service so the Go layer can call it. Engine logic stays synchronous and pure; only the thin HTTP layer is async. Intended endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/functions` | Parse, lower, and optimize an expression; return an id and its variables |
| `POST` | `/eval` | Evaluate a compiled function at a point |
| `POST` | `/grad` | Full gradient at a point |
| `POST` | `/jacobian` | Jacobian matrix at a point |
| `POST` | `/trace` | Forward/backward trace for the visualizer |
| `POST` | `/solve` | Run a solver; return per-iteration history (streamed later via the Go WebSocket layer) |

Compiled functions are held in an in-memory `id -> Graph` map behind a lock. Request and response bodies are `serde` structs.

## Documentation and roadmap

- **[`roadmap.md`](roadmap.md)**: the authoritative source. Product summary, background math, the Rust learning track, tech-stack rationale, architecture, full file structure, and every phase and ticket in detail. This README is a snapshot; the roadmap remains the working plan until the project is complete.

## Roadmap and status

Confidence tiers (Tier 1 is the real target; the rest is bonus):

- **Tier 1, resume-ready:** Phases 0 through 3 (Rust warmup, autodiff core, Jacobian, parser) plus one solver, a minimal demo, and a benchmark and this README. A complete, standalone, all-Rust compiler and autodiff engine.
- **Tier 2, the full system:** the Go service layer, Temporal durability, crash/resume, and the animated graph visualizer.
- **Tier 3, polish:** the IK arm canvas, a convergence plot, extra optimization passes, damped least squares, and Postgres.

Phase progress:

- [x] **Phase 0**: Rust warmup and crate skeleton
- [~] **Phase 1**: Autodiff core (arena, ops, forward, backward, finite-diff oracle) implemented; Jacobian done
- [~] **Phase 2**: Extend the math (full op set + errors + Jacobian done; JSON trace emission pending)
- [ ] **Phase 3**: Compiler front end (lexer, Pratt parser, AST-to-graph lowering)
- [ ] **Phase 4**: Optimization passes (constant folding, CSE, dead-code elimination)
- [ ] **Phase 4.5**: Symbolic differentiation and higher-order derivatives (the contrast piece)
- [ ] **Phase 5**: Solvers (LU, Newton, inverse kinematics)
- [ ] **Phase 6**: Rust engine as an HTTP service
- [ ] **Phase 7 and 8**: Go service layer + Temporal durability (Tier 2)
- [ ] **Phase 9**: TypeScript visualizer (Tier 2/3)
- [ ] **Phase 10**: Benchmarks and final README

The two headline benchmarks to come: reverse-vs-forward differentiation cost as input dimension grows (reverse stays flat, the others grow linearly, which is why backprop scales), and graph node count before versus after the optimization passes.

---

Solo project by Yulchan, built partly as a way to learn Rust on its hardest idiomatic case: the arena graph.
