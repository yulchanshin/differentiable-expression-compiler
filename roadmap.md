# Gradient Engine — Product Roadmap (Rust-Engine Edition)

> A differentiable expression compiler & durable optimization service. Rust engine (the compiler + automatic differentiation + solvers), Go service layer, TypeScript visualizer. **Sequenced math-first so you learn Rust on the interesting part.**

**Duration:** ~3–5 weeks (Rust learning included) · **Owner:** Yulchan · **Last updated:** July 2026

---

## Table of Contents

1. [Product Summary](#1-product-summary)
2. [How This Roadmap Is Sequenced](#2-how-this-roadmap-is-sequenced)
3. [Scope, Realism & Confidence Tiers](#3-scope-realism--confidence-tiers)
4. [Background Knowledge (the math + concepts)](#4-background-knowledge)
5. [Rust Learning Track](#5-rust-learning-track)
6. [Tech Stack](#6-tech-stack)
7. [Architecture](#7-architecture)
8. [File Structure](#8-file-structure)
9. [Progress Overview](#9-progress-overview)
10. [Phases & Tickets](#10-phases--tickets)
11. [Testing & Benchmarks](#11-testing--benchmarks)
12. [Resume Framing](#12-resume-framing)

---

## 1. Product Summary

**Gradient Engine** takes a math function over several variables — e.g. `f(x, y) = sin(x*y) + x^2` — compiles it into a computational graph (a DAG), and can then evaluate it, differentiate it *exactly* via reverse-mode automatic differentiation (full gradient + Jacobian), and drive durable iterative solvers (Newton's method, inverse kinematics) on top of it. A live browser visualizer animates the differentiation happening on the actual graph, plus an IK arm that reaches toward a clicked target.

**Why it reads as SWE, not ML.** The bulk of the code is a real compiler pipeline — lexer → parser → graph IR → optimization passes → execution — plus a durable-orchestration layer and a full-stack visualizer. The calculus is the *payload*; the *substance* is PL/compilers + distributed systems + full-stack. It's deliberately aimed at solvers and a robot arm rather than neural nets.

**The Rust angle.** The entire engine — compiler front end, automatic differentiation, optimizer, and solvers — is written in **Rust**, which you're learning as you build. The Go service layer and TypeScript frontend surround it. The language boundary sits at a genuine *compute-vs-serving* seam (the Rust engine is its own service the Go layer calls), which is a normal, defensible architecture — not an artificial split through the middle of one component.

**The one mental model to hold onto:** *A computational graph is a DAG. The forward pass is a topological-order evaluation. The backward pass is a reverse-topological-order traversal that accumulates adjoints.* Everything hangs off that spine.

**The intellectual payload (your README headline):** reverse-mode AD computes the *entire* gradient — every partial — in a single backward pass at the cost of ~one function evaluation, **regardless of the number of inputs**. Forward-mode and finite differences cost one pass *per input*. That asymmetry is why backprop scales, and demonstrating it (benchmark + short derivation) is the "cool math, understood" proof.

---

## 2. How This Roadmap Is Sequenced

**Math-first, parser-last. Learn Rust on the graph, not the lexer.**

The classic instinct is to build a compiler front-to-back: lexer → parser → graph → autodiff. **This roadmap deliberately does not.** Because you're learning Rust as you go, the order is chosen so you meet Rust's hard parts on the *core data structure* (the graph) while building the *interesting* part (the calculus), and only add the text-parsing front end once you're already comfortable.

The sequence:

1. **Rust warmup** (Phase 0) — tiny exercises to get syntax, ownership, and `cargo` under your fingers before touching the real code.
2. **Autodiff core on hand-built graphs** (Phase 1) — you construct `f(x,y) = sin(x*y)+x²` *by manually wiring nodes in Rust*, no parser. Forward eval, then the reverse-mode backward pass, validated against finite differences. **This is where the project becomes real and where you learn the arena-graph pattern — the single most important Rust lesson here.**
3. **Extend the math** (Phase 2) — more operations, Jacobians, and the trace data the visualizer will later animate.
4. **Now the compiler front end** (Phase 3) — lexer + Pratt parser that produces the *same graph you've been building by hand*. By now you know Rust and you know the target, so this is low-stakes.
5. **Optimizer, solvers, service** (Phases 4–6) — the rest of the Rust engine.
6. **Go services, Temporal, frontend, benchmarks** (Phases 7–10) — everything around the engine.

**Why this is better for you specifically:** you get a working gradient engine (the cool part) in the first week or so, you learn Rust's ownership model on the DAG where it's genuinely challenging (rather than dodging it), and the parser — normally the intimidating "start" of a compiler — becomes a comfortable mid-project task instead of a Rust-and-compilers double-whammy on day one.

**Each ticket carries a 🦀 "Rust concepts introduced" block** so the language learning is paced and mapped to real work, never abstract. Read [§5 Rust Learning Track](#5-rust-learning-track) for the overall concept progression before you start.

---

## 3. Scope, Realism & Confidence Tiers

**Timeline reality with Rust-from-scratch included.** The all-Rust engine — specifically the graph (a multi-parent DAG) — is Rust's hard case, so budget more than a pure-Go build would take. Honest estimate, heavy AI use, part-time around your internship:

- **Rust warmup + autodiff core on hand-built graphs (Phases 0–2): the meat of the learning.** Slower at first as Rust ownership clicks; expect the arena-graph and backward-pass tickets to take real time. But this is where you *learn Rust*, so the time is the point, not waste.
- **Compiler front end (Phase 3): faster than it looks** once you know Rust — parsing a tree is Rust's *easy* case (single ownership).
- **Solvers + service (Phases 4–6): moderate.**
- **Go services + Temporal + frontend (Phases 7–10): as before**, unaffected by the Rust decision except that Go now calls the Rust engine over a wire.

**Confidence tiers** — treat **Tier 1 as the real deadline**; everything above is bonus you add while already applying.

| Tier | Target | Confidence | Contents |
|---|---|---|---|
| **Tier 1 — resume-ready** | ~end of week 2–3 | **High** | Phases 0–3 (Rust warmup + autodiff core + Jacobian + parser) + one solver + a minimal visualizer + README with finite-diff validation and the reverse-vs-forward benchmark. A complete, cool, **all-Rust compiler + autodiff engine**. If you stop here, you've won *and* you've learned Rust. |
| **Tier 2 — the full system** | week 3–4 | **Good** | Add the Go service layer + Temporal durability + crash/resume + the animated graph visualizer. This makes it a distributed full-stack system. |
| **Tier 3 — polish** | ongoing while applying | **Nice-to-have** | IK arm canvas, convergence plot, extra optimization passes, damped least squares, Postgres. |

**Cut order if short on time:** Postgres → extra optimization passes → DLS (ship Jacobian-transpose IK only) → convergence plot → IK arm. **Never cut:** the autodiff core, the parser, and (Tier 2) Temporal + the graph visualizer.

> Note the shape shift from the Go plan: here, **Tier 1 is itself a complete Rust project** (engine + compiler, no Go needed). The Go/Temporal/frontend work is genuinely Tier 2+. This means even a "half-finished" outcome is a finished, standalone, impressive thing.

---

## 4. Background Knowledge

Skim in this order. None of it is combinatorics-hard; the concepts are self-contained.

### 4.1 Automatic differentiation (the core idea)

AD is a **third thing**, distinct from:
- **Symbolic differentiation** (Wolfram): manipulates formulas into formulas. Exact but expressions blow up.
- **Numerical differentiation** (finite differences): `f'(x) ≈ (f(x+h) − f(x)) / h`. Approximate, one eval per input.
- **Automatic differentiation**: applies the chain rule *locally* at each primitive op over the graph. **Exact**, at ~one function-evaluation's cost.

Two modes: **forward** (cost scales with #inputs) and **reverse** (cost scales with #outputs). You're building **reverse mode** — the many-inputs-few-outputs case, which is also backprop.

> **Read:** Karpathy's `micrograd` source (~150 lines, the whole concept in miniature — ignore the neural-net framing, study the engine). The "Automatic differentiation" Wikipedia article's forward-vs-reverse sections.

### 4.2 The multivariable chain rule (the calculus)

The backward pass *is* the multivariable chain rule over a DAG:
- **Adjoint** of a node = ∂(output)/∂(node) — how much the output moves per unit change in this node.
- Seed the output's adjoint to **1** (∂f/∂f = 1).
- At each node, push its adjoint to its inputs by multiplying by the op's **local derivative**.
- A node with multiple parents **accumulates** (`+=`) contributions from every path — the sum-over-paths rule. This is why shared subexpressions matter and why you accumulate rather than assign.

### 4.3 Local derivative table (the mathematical contract)

| Op | Forward `v` | Adjoint to inputs (incoming adjoint `ḡ`) |
|---|---|---|
| `add(a,b)` | `a+b` | `ā += ḡ`; `b̄ += ḡ` |
| `sub(a,b)` | `a−b` | `ā += ḡ`; `b̄ += −ḡ` |
| `mul(a,b)` | `a*b` | `ā += ḡ*b`; `b̄ += ḡ*a` |
| `div(a,b)` | `a/b` | `ā += ḡ/b`; `b̄ += −ḡ*a/(b*b)` |
| `pow(a,k)` | `a^k` | `ā += ḡ*k*a^(k−1)` |
| `sin(a)` | `sin a` | `ā += ḡ*cos a` |
| `cos(a)` | `cos a` | `ā += ḡ*(−sin a)` |
| `exp(a)` | `exp a` | `ā += ḡ*exp a` (= `ḡ*v`) |
| `ln(a)` | `ln a` | `ā += ḡ/a` |
| `neg(a)` | `−a` | `ā += −ḡ` |

Note the recurring "value forward, adjoint backward" shape — the push/accumulate invariant.

### 4.4 Jacobians

For `f: ℝⁿ → ℝᵐ`, the Jacobian `J` is the m×n matrix `J[i][j] = ∂fᵢ/∂xⱼ`. Reverse mode gives one *row* per backward pass (seed output i's adjoint to 1, rest 0). Your solvers consume J.

### 4.5 Newton's method

Solve `f(x)=0`: (1) eval `f(x)` and Jacobian `J`; (2) solve `J·Δx = −f(x)`; (3) `x ← x + Δx`; repeat until `‖f(x)‖` small. Needs a small dense linear solver (LU with partial pivoting).

### 4.6 Inverse kinematics (flagship demo)

A 2D jointed arm: end-effector position `p(θ)` is a function of joint angles. The Jacobian `J = ∂p/∂θ` (your engine gives it free) drives the tip to a target. **Jacobian-transpose** `Δθ = α·Jᵀ(t−p)` is the simple start; **damped least squares** `Δθ = Jᵀ(JJᵀ + λ²I)⁻¹(t−p)` is the smoother upgrade.

> **Read:** Buss, *"Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares."*

### 4.7 Compiler concepts (Phase 3+)

- **Lexer:** text → tokens.
- **Pratt parsing:** hand-written operator-precedence parser (why `x^2^3` groups right, `a+b*c` groups the multiply first).
- **AST → graph lowering** with **hash-consing** (dedup identical subexpressions so shared work is a shared node).
- **Topological sort:** forward-eval order; reverse it for backward.
- **Optimization passes:** constant folding, CSE, dead-code elimination.

> **Read:** Bob Nystrom, *"Pratt Parsers: Expression Parsing Made Easy"*; the Pratt chapter of *Crafting Interpreters*.

### 4.8 Temporal & WebSocket (Phase 7+)

- **Temporal** (Go SDK): durable workflows + activities; state checkpointed via event history, so a crashed worker replays and resumes. Your solver loop becomes a workflow.
- **WebSocket:** real-time JSON frames from Go to the browser (AD trace + solver progress).

---

## 5. Rust Learning Track

This is the concept progression, mapped to where each idea first appears in a real ticket. **Don't pre-study all of it** — learn each concept when its ticket demands it. But read this once so you see the arc.

### Prerequisite Rust (before TICKET-000)

Do a *small, focused* amount of Rust before starting — but **don't fall into the "learn Rust first, then build" trap**, which quietly becomes three weeks of tutorials and no project. The goal of pre-study is narrow: get the *syntax* automatic so your brain is free for the borrow checker and the graph when you start. Everything else you learn on the project, at the ticket that needs it.

**Learn before you start (~6–10 hours total):**

| Session | Time | Content | Why |
|---|---|---|---|
| 1 | ~3 hrs | *The Rust Book* ch. 1–3 (setup, syntax, types, functions, control flow). Move fast. Get `cargo` working; write and run a few functions. | Remove syntax friction so it's not competing for attention later. |
| 2 | ~3 hrs | *The Rust Book* **ch. 4 (ownership & borrowing) — read slowly, twice** — then ch. 5 (structs) + ch. 6 (enums + `match`). Do the matching `rustlings` sections. | Ch. 4 is the concept the whole project leans on; ch. 5–6 are the exact tools for TICKET-100/101. |
| 3 | ~2 hrs | Skim two things: "arena / index-based graph in Rust," and **"why `Rc<RefCell<T>>` exists and why it's often a code smell."** Then start TICKET-000. | So you've *pre-decided* to use indices, not `Rc<RefCell>`, before the internet/AI pushes you toward the bait. |

**Pair reading with writing.** Reading Rust and writing Rust are different skills — the borrow checker only teaches by rejecting your code. Use **`rustlings`** (the standard on-ramp; do through "enums" and "error handling") or the TICKET-000 warmup exercises. Treat "Book ch. 1–6 + rustlings" as true day-zero, and TICKET-000 as the bridge into real work.

**The non-obvious prerequisite:** the 20 minutes on *why not `Rc<RefCell>`* matters more than it looks. When you get stuck on the graph in TICKET-100, every search result and AI suggestion will push `Rc<RefCell<Node>>` as the "easy fix." Knowing in advance that you're deliberately using an **arena of indices** instead — and why — is what keeps you out of the miserable beginner path. Knowing what you're *not* doing is as valuable as knowing what you are.

**Everything else: learn on the project.** `Result`/`?` (TICKET-104/200), `Box` for recursive enums (TICKET-301), `serde`/closures/`HashMap`/traits/async — each is introduced at its ticket in the progression below. Pre-studying them cold is inefficient; they stick when tied to concrete use.

> **Mindset note:** Rust punishes "submit first, diagnose later" at *compile* time rather than runtime — which feels different, but it's the same interactive-debugging loop you already like, just moved earlier. Reframe borrow-checker errors as a strict pair-programmer: each one points at a real ownership question. Use the roadmap's loop — write it yourself, hit the error, ask AI to *explain* (not just fix) it, apply the fix yourself. That turns each rejection into a 5-minute lesson instead of a wall.

### The progression (each builds on the last)

| Stage | Concepts | First appears in |
|---|---|---|
| **A — Basics** | `cargo`, `let`/`mut`, functions, primitive types (`f64`, `usize`, `bool`), `println!`/`dbg!`, `if`/`loop`/`for`, `Vec<T>`, `String` vs `&str` | TICKET-000 (warmup) |
| **B — Ownership & borrowing** | move semantics, `&` (shared borrow), `&mut` (exclusive borrow), the borrow-checker's one-mutable-XOR-many-shared rule, why this exists | TICKET-000 → TICKET-100 |
| **C — Structs & enums** | `struct`, `enum` with data, `match` (exhaustive), `derive(Debug, Clone)` | TICKET-100, TICKET-101 |
| **D — The arena pattern** | representing a graph as `Vec<Node>` + `usize` indices instead of pointers/`Rc`; **why this is the idiomatic Rust answer to shared-ownership graphs** | TICKET-100 (⭐ the key lesson) |
| **E — Iteration & mutation** | iterating a `Vec`, indexing, iterating in reverse, mutating through indices to sidestep the borrow checker | TICKET-102, TICKET-103 |
| **F — Error handling** | `Option<T>`, `Result<T, E>`, the `?` operator, custom error `enum`s, `panic!` vs recoverable errors | TICKET-104, TICKET-200 |
| **G — Closures & generics (light)** | `Fn` closures (for the finite-diff oracle), basic generics `<T>`, `impl` blocks & methods | TICKET-104 |
| **H — Recursion & `Box`** | `Box<T>` for recursive types (the AST enum), recursive functions over a tree | TICKET-301 (parser) |
| **I — Traits** | `trait`, `impl Trait for Type`, `Display`/`Debug`, using traits for op dispatch | TICKET-200 / TICKET-301 |
| **J — Collections** | `HashMap` (hash-consing, variable name→index maps) | TICKET-302 |
| **K — Serialization** | `serde` + `serde_json` for the trace/API boundary | TICKET-205 / TICKET-600 |
| **L — Async & web (Tier 2)** | `tokio`, an HTTP framework (`axum`), async/await basics, `Result` in handlers | TICKET-600 |

### The one lesson that matters most: **the arena pattern (Stage D)**

Rust makes pointer-based graphs painful *on purpose* — a node with two parents has two owners, which violates single-ownership, and the beginner workaround (`Rc<RefCell<Node>>`) is miserable. **The idiomatic answer is an arena:** store all nodes in one `Vec<Node>`, and let nodes reference each other by their **index** (`usize`) in that vector, not by pointer. The `Vec` owns everything; indices are just numbers you copy freely. This makes the borrow checker mostly leave you alone, and it's genuinely how real Rust compilers and graph libraries work.

You'll internalize this in TICKET-100, and once it clicks, the rest of the engine stops fighting you. If you learn *one* Rust idea deeply from this project, make it this one — it's transferable to every graph/tree/AST you ever build in Rust.

### How to use AI while learning (so you actually learn)

- **Do write yourself:** the graph representation, the backward pass, the op derivatives. These are the learning. If AI writes them, you learn nothing and can't debug them.
- **Let AI draft:** boilerplate (`Cargo.toml`, error-enum plumbing, serde derives, the HTTP handler skeleton), and *explanations* of borrow-checker errors ("why won't this compile?").
- **Best pattern:** write it yourself, hit a borrow-checker error, paste the error to AI and ask it to *explain* (not just fix) — then apply the fix yourself. That loop teaches you Rust's model instead of laundering around it.
- **Red flag:** if you're pasting AI-generated Rust you don't understand into the *engine core*, stop — that's the exact failure mode to avoid. Understanding the core is the point.

---

## 6. Tech Stack

| Layer | Technology | Role | Notes |
|---|---|---|---|
| **Engine** (compiler + autodiff + solvers) | **Rust** | Lexer, Pratt parser, graph IR, forward/reverse AD, optimizer, LU, Newton, IK. Runs as its own service. | The whole computational core; where you learn Rust |
| Engine crates | `serde`/`serde_json` (serialization), `axum` + `tokio` (HTTP, Tier 2) | Expose the engine over HTTP/JSON | Added only when the service ticket needs them |
| Engine ↔ Go transport | **HTTP/JSON** (or gRPC) | Go calls the Rust engine as a service | HTTP is less setup; gRPC is a better story if you want it |
| Service / orchestration (BFF) | **Go** (`net/http` + `websocket`) | REST, WebSocket streaming, Temporal client, Postgres | Tier 2 |
| Durability | **Temporal (Go SDK)** | Durable, resumable solver workflows | Tier 2 |
| Persistence | **Postgres** (`sqlc` or `pgx`, no ORM) | Saved functions, run history | Tier 3, first to cut |
| Frontend / visualizer | **TypeScript + React + Vite**, **D3 + dagre**, Canvas | Animated AD graph, IK arm, convergence plot | Tier 2/3 |
| Browser streaming | **WebSocket** | Go ↔ browser real-time frames | Tier 2 |
| Infra | **Docker Compose** | Rust engine + Go server + Temporal + Postgres locally | |

**The split rationale (interview-ready):** Rust owns the engine because that's where exact, fast numerical computation and the compiler live — the part worth the borrow-checker investment. Go owns orchestration because Temporal is Go-native and Go excels at concurrent service glue. The boundary is a real compute-vs-serving seam, so it's a sound architecture, not a language gimmick.

---

## 7. Architecture

### 7.1 Component diagram

```
                    ┌───────────────────────────────────────────────┐
                    │                 BROWSER (SPA)                  │
                    │  TypeScript + React + Vite                     │
                    │  Graph Visualizer (D3+dagre) · IK Arm · Plot   │
                    └───────────────┬───────────────────────────────┘
                                    │  WebSocket (JSON frames) + REST
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │                  GO SERVICE LAYER (BFF)  [Tier 2]          │
        │  REST · WebSocket · Temporal client · Postgres            │
        └───────┬──────────────────────┬───────────────────┬────────┘
                │ HTTP/JSON (or gRPC)   │ Temporal          │ SQL
                ▼                       ▼                   ▼
   ┌────────────────────────┐  ┌──────────────────┐  ┌──────────────┐
   │  RUST ENGINE (service) │  │  TEMPORAL SERVER │  │  POSTGRES    │
   │  ── the whole compiler │  │  durable solver  │  │  functions,  │
   │  lexer → parser →      │  │  workflows       │  │  runs        │
   │  graph IR → optimizer  │  │                  │  │              │
   │  → forward/reverse AD  │  └──────────────────┘  └──────────────┘
   │  → LU/Newton/IK        │
   └────────────────────────┘
        ▲
        │  Tier 1 stops here: the Rust engine alone (callable via CLI
        │  or a thin HTTP endpoint) is already a complete project.
```

### 7.2 Tier 1 is standalone

Crucially, the Rust engine does **not** need Go to be a finished, demoable project. In Tier 1 you can exercise it via a small CLI or a single HTTP endpoint and show gradients, Jacobians, and a solver working. Go, Temporal, and the browser viz are Tier 2+ additions that turn a great *engine* into a great *system*.

### 7.3 The engine ↔ frontend trace contract (build early, in Rust)

The visualizer animates whatever the engine emits. Design the trace as a first-class engine output (a Rust struct that `serde`-serializes to this JSON), not a retrofit:

```jsonc
{
  "graph": {
    "nodes": [
      { "id": 0, "op": "var", "label": "x" },
      { "id": 1, "op": "var", "label": "y" },
      { "id": 2, "op": "mul", "inputs": [0, 1] },
      { "id": 3, "op": "sin", "inputs": [2] },
      { "id": 4, "op": "pow", "inputs": [0], "attr": { "k": 2 } },
      { "id": 5, "op": "add", "inputs": [3, 4] }
    ],
    "output": 5
  },
  "forward":  [ { "id": 0, "value": 1.5 }, /* … topological order … */ ],
  "backward": [ { "id": 5, "adjoint": 1.0 }, /* … reverse-topo order … */ ]
}
```

Two ordered arrays over a shared node list — all the frontend needs to animate step-by-step.

---

## 8. File Structure

```
gradient-engine/
├── README.md                       # headline result + benchmark + the derivation
├── docker-compose.yml              # rust-engine, go-server, temporal, postgres  [Tier 2]
├── Makefile
│
├── engine/                         # ← RUST. The whole compiler + AD + solvers.
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # CLI entry (Tier 1) / server entry (Tier 2)
│       ├── lib.rs                  # crate root, re-exports
│       ├── graph/
│       │   ├── mod.rs
│       │   ├── node.rs             # Node struct, OpType enum
│       │   ├── arena.rs            # ⭐ Vec<Node> arena + index-based construction
│       │   └── topo.rs             # topological sort
│       ├── autodiff/
│       │   ├── mod.rs
│       │   ├── forward.rs          # forward evaluation
│       │   ├── backward.rs         # ⭐ reverse-mode backward pass
│       │   ├── jacobian.rs
│       │   └── trace.rs            # emit the §7.3 trace (serde)
│       ├── ops/
│       │   └── derivatives.rs      # per-op forward + local-derivative table
│       ├── optimize/
│       │   ├── constfold.rs
│       │   ├── cse.rs
│       │   └── deadcode.rs
│       ├── parse/                  # ← Phase 3, AFTER the core works
│       │   ├── lexer.rs
│       │   ├── ast.rs              # AST enum (Box-recursive)
│       │   ├── parser.rs           # Pratt parser
│       │   └── lower.rs            # AST → graph (hash-consing)
│       ├── linalg/
│       │   └── lu.rs               # LU decomposition
│       ├── solver/
│       │   ├── newton.rs
│       │   └── ik.rs
│       ├── error.rs                # custom error enum
│       └── api/                    # ← Tier 2: HTTP handlers (axum)
│           └── http.rs
│   └── tests/
│       └── finite_diff.rs          # ⭐ the correctness oracle
│
├── server/                         # ← GO. Service layer / BFF.  [Tier 2]
│   ├── go.mod
│   ├── main.go
│   ├── engineclient/               # HTTP/gRPC client to the Rust engine
│   ├── rest/
│   ├── ws/                         # WebSocket: trace + solver streams
│   └── store/                      # Postgres (sqlc/pgx)
│
├── worker/                         # ← GO. Temporal workflows + activities.  [Tier 2]
│   ├── go.mod
│   ├── main.go
│   ├── workflow/solve_workflow.go
│   └── activity/step_activity.go
│
├── web/                            # ← TypeScript + React + Vite.  [Tier 2/3]
│   └── src/
│       ├── api/                    # rest.ts, ws.ts
│       ├── types/trace.ts          # mirror of the §7.3 contract
│       └── components/             # GraphViz, IKArm, ConvergencePlot, FunctionInput
│
└── bench/
    ├── reverse_vs_forward.rs       # cost vs input dimension  (a Rust bench)
    └── results/
```

---

## 9. Progress Overview

### Phase 0 — Rust warmup & setup
- [X] `TICKET-000` Rust toolchain + warmup exercises
- [X] `TICKET-001` Engine crate skeleton + workspace

### Phase 1 — Autodiff core on hand-built graphs ⭐ (the math, and the key Rust learning)
- [X] `TICKET-100` Arena graph representation (`Vec<Node>` + indices)
- [X] `TICKET-101` `OpType` enum + hand-construction helpers
- [X] `TICKET-102` Forward evaluation (topological order)
- [X] `TICKET-103` Reverse-mode backward pass (the AD core)
- [X] `TICKET-104` Finite-difference oracle (validation harness) ### Phase 2 — Extend the math - [ ] `TICKET-200` Full op set + local-derivative table + error handling [ ] `TICKET-201` Jacobian (multi-output) [ ] `TICKET-205` Trace emission (frontend contract, serde) ### Phase 3 — Compiler front end (now that Rust + graph are solid) [ ] `TICKET-300` Lexer [ ] `TICKET-301` Pratt parser → AST
- [ ] `TICKET-302` AST → graph lowering (hash-consing)

### Phase 4 — Optimization passes
- [ ] `TICKET-400` Constant folding
- [ ] `TICKET-401` Common-subexpression elimination
- [ ] `TICKET-402` Dead-node elimination + node-count benchmark

### Phase 5 — Solvers
- [ ] `TICKET-500` LU linear solver
- [ ] `TICKET-501` Newton's method
- [ ] `TICKET-502` IK solver (Jacobian-transpose → damped least squares)

### Phase 6 — Rust engine as a service
- [ ] `TICKET-600` Expose engine over HTTP/JSON (axum + serde)

### Phase 7 — Go service layer (BFF) · Tier 2
- [ ] `TICKET-700` Go server + engine client + REST
- [ ] `TICKET-701` WebSocket streaming (trace + solver)
- [ ] `TICKET-702` Postgres store · Tier 3

### Phase 8 — Durability (Temporal, Go) · Tier 2
- [ ] `TICKET-800` Temporal worker skeleton
- [ ] `TICKET-801` Solve workflow + crash/resume demo

### Phase 9 — Frontend · Tier 2/3
- [ ] `TICKET-900` App shell + REST wiring
- [ ] `TICKET-901` Graph visualizer (animated AD)
- [ ] `TICKET-902` IK arm canvas · Tier 3
- [ ] `TICKET-903` Convergence plot · Tier 3

### Phase 10 — Benchmarks & README
- [ ] `TICKET-1000` Reverse-vs-forward benchmark
- [ ] `TICKET-1001` README: architecture, results, derivation

**Tier 1 = Phases 0–3 + one solver (500/501 or 502) + a minimal way to show it + TICKET-1000/1001.** That alone is a finished, standalone, all-Rust compiler + autodiff engine.

---

## 10. Phases & Tickets

Every ticket has: number, title, branch, description, detail, acceptance criteria, "learn/read," and — for Rust tickets — a 🦀 **Rust concepts introduced** block that paces your language learning against real work.

**Golden rule (repeated from §5):** write the engine core yourself; use AI to *explain* borrow-checker errors and draft boilerplate, never to write the graph/AD logic you're supposed to be learning.

---

### PHASE 0 — Rust warmup & setup

---

#### TICKET-000 — Rust toolchain + warmup exercises
**Branch:** `chore/000-rust-warmup`

**Description:** Install Rust, get `cargo` under your fingers, and do small throwaway exercises so you meet ownership/borrowing on trivial code *before* the real graph. Do **not** skip this — 2–4 hours here saves days later.

**Detail:**
- Install via `rustup`; confirm `cargo --version`, `rustc --version`.
- `cargo new warmup` and work through these micro-exercises (each is a `fn` you write and test with `dbg!`/`assert_eq!`):
  1. A function that sums a `Vec<f64>` (learn: `Vec`, `for`, references `&`).
  2. A function that takes `&mut Vec<f64>` and doubles every element in place (learn: `&mut`, mutation).
  3. An `enum Shape { Circle(f64), Rect(f64,f64) }` with an `area()` via `match` (learn: enums with data, exhaustive `match`).
  4. Deliberately trigger a borrow-checker error (e.g. hold a `&` and a `&mut` to the same `Vec` at once), read the error, and understand *why* it's rejected. **This is the single most valuable warmup — meet the borrow checker on purpose.**
- Recommended: first ~4 chapters of *The Rust Book* (rust-lang.org/book) or `rustlings` exercises for stages A–C.

**Acceptance criteria:**
- [X] `cargo build` / `cargo test` work.
- [X] You can explain, in a sentence, why exercise 4 fails to compile (one mutable XOR many shared borrows).
- [X] The four exercises pass their asserts.

🦀 **Rust concepts introduced:** Stage A (cargo, `let`/`mut`, functions, `Vec`, `for`, `println!`/`dbg!`) and first contact with Stage B (`&` vs `&mut`, the borrow rule) and Stage C (`enum`, `match`).

**Learn/read:** *The Rust Book* ch. 1–4 (ownership is ch. 4 — read it slowly, twice); `rustlings`.

---

#### TICKET-001 — Engine crate skeleton + workspace
**Branch:** `chore/001-crate-skeleton`

**Description:** Create the real `engine` Rust crate with the module layout from §8 (empty modules with doc comments). Set up the workspace so Go modules can be added later without disruption.

**Detail:**
- `cargo new engine --lib` (library crate; `main.rs` is a thin CLI on top later).
- Create empty `mod` files matching §8 (`graph`, `autodiff`, `ops`, `optimize`, `parse`, `linalg`, `solver`, `error`) with `//! module doc` headers.
- Add a `tests/` dir. Add `Makefile` targets `test`, `run`, `bench`.
- Repo root holds the Go/`web` dirs later; keep the Rust crate self-contained under `engine/`.

**Acceptance criteria:**
- [X] `cargo build` and `cargo test` pass on the empty skeleton.
- [X] Module tree matches §8; each module compiles.

🦀 **Rust concepts introduced:** crate vs module (`mod`, `pub`, `use`), library vs binary crate, `lib.rs` as crate root, doc comments (`//!`, `///`).

**Learn/read:** *The Rust Book* ch. 7 (packages, crates, modules).

---

### PHASE 1 — Autodiff core on hand-built graphs ⭐

> This phase is the heart of both the project and your Rust learning. You build the graph and the calculus **by hand-constructing nodes in Rust** — no parser yet. By the end you have a working reverse-mode autodiff engine validated against finite differences. Take your time here; the arena pattern you learn in TICKET-100 is the key that unlocks everything after.

---

#### TICKET-100 — Arena graph representation (`Vec<Node>` + indices) ⭐
**Branch:** `feat/100-arena-graph`

**Description:** The most important Rust decision in the project. Represent the computation DAG as a single `Vec<Node>` arena where nodes reference each other by **index** (`usize`), not by pointer or `Rc`. This is what makes a multi-parent graph tractable in Rust.

**Detail:**
- Define `struct Graph { nodes: Vec<Node> }`.
- Define `struct Node { op: OpType, inputs: Vec<usize>, value: f64, adjoint: f64 }` (fill `OpType` in TICKET-101; a placeholder enum is fine now).
- Add `impl Graph`: a `new()`, and a `push(&mut self, node: Node) -> usize` that appends and returns the new node's index. That returned index is your "node handle."
- **Understand deeply why this beats `Rc<RefCell<Node>>`:** every node lives in one owner (the `Vec`); indices are `Copy` numbers you pass around freely; a node with two parents is just an index appearing in two `inputs` lists — no ownership conflict. Write a comment in `arena.rs` explaining this in your own words (it cements the lesson and is great interview prep).

**Acceptance criteria:**
- [X] You can build a graph for `x * y` by hand: push two var nodes, then a `mul` node referencing their indices; assert the structure.
- [X] A shared node (`x` used twice) is a single index appearing in two inputs lists — demonstrated in a test.
- [X] `arena.rs` contains your own-words explanation of why indices beat pointers here.

🦀 **Rust concepts introduced (⭐ Stage D — the big one):** the arena pattern; `usize` indices as handles; why `Copy` types (indices) sidestep ownership; `struct` with fields; `Vec<T>` as an owner; `&mut self` methods; `derive(Debug, Clone)`. This is the ticket where Rust's ownership model *clicks* for graphs.

**Learn/read:** search "Rust arena pattern graph" / "Rust index-based graph" (the "Modeling graphs in Rust using vector indices" style write-ups); why `Rc<RefCell>` is discouraged for this. *The Rust Book* ch. 5 (structs), ch. 6 (enums).

---

#### TICKET-101 — `OpType` enum + hand-construction helpers
**Branch:** `feat/101-op-enum`

**Description:** Define the operation types as a Rust `enum`, and add ergonomic helpers to build graphs by hand so writing test expressions isn't painful.

**Detail:**
- `enum OpType { Var(String), Const(f64), Add, Sub, Mul, Div, Neg, Pow(f64), Sin, Cos, Exp, Ln }` — note data-carrying variants (`Var`, `Const`, `Pow` hold data).
- Builder helpers on `Graph`: `var(&mut self, name)`, `constant(&mut self, v)`, `add(&mut self, a, b)`, `mul(&mut self, a, b)`, `sin(&mut self, a)`, etc. — each pushes a node and returns its index. These let you write:
  ```rust
  let x = g.var("x");
  let y = g.var("y");
  let xy = g.mul(x, y);
  let s = g.sin(xy);
  let x2 = g.pow(x, 2.0);
  let f = g.add(s, x2);      // f(x,y) = sin(x*y) + x^2, built by hand
  ```
- Track which index is an input variable (name → index map, or a `Vec` of var indices).

**Acceptance criteria:**
- [X] The `sin(x*y) + x^2` graph builds with the helpers in a few readable lines.
- [X] `match`ing on `OpType` is exhaustive (compiler enforces it — lean on that).
- [X] Test asserts the node count and structure of a couple of hand-built expressions.

🦀 **Rust concepts introduced (Stage C deepened):** `enum` variants that carry data (`Pow(f64)`, `Var(String)`); exhaustive `match` and how the compiler *forces* you to handle every op (a feature — it catches missing cases); method chaining/builder ergonomics; `String` ownership in `Var`.

**Learn/read:** *The Rust Book* ch. 6 (enums + `match`) — the `Option` examples map directly onto op dispatch.

---

#### TICKET-102 — Forward evaluation (topological order)
**Branch:** `feat/102-forward-eval`

**Description:** Evaluate the hand-built graph at given variable values by filling each node's `value` in topological order.

**Detail:**
- Topological sort over the arena (Kahn's algorithm, or exploit that if you always push inputs before their consumers, index order *is* a valid topo order — a nice property of build-order construction; note it and rely on it, or sort explicitly for safety).
- `fn forward(&mut self, inputs: &HashMap<String, f64>) -> f64`: iterate nodes in topo order, compute each node's `value` by `match`ing its `OpType` and reading its inputs' already-computed `value`s. Return the output node's value.
- Guard: `Var` reads from the inputs map; error if missing (return `Result` — see TICKET-104/200, or `panic!` for now and upgrade later).

**Acceptance criteria:**
- [X] `f(x,y)=sin(x*y)+x^2` at `(1.5, 2.0)` matches a hand/Python-computed value to 1e-9.
- [X] A shared subexpression is computed once, not twice.

🦀 **Rust concepts introduced (Stage E):** iterating a `Vec` by index; reading one element while the loop proceeds; `match` returning values; `HashMap<String,f64>` lookups (`.get()` returns `Option`); `f64` methods (`.sin()`, `.powf()`, `.exp()`, `.ln()`).

**Learn/read:** Kahn's algorithm; Rust `HashMap` basics (*Rust Book* ch. 8).

---

#### TICKET-103 — Reverse-mode backward pass (the AD core) ⭐
**Branch:** `feat/103-backward-pass`

**Description:** The intellectual center. After a forward pass, compute every input variable's partial derivative in one backward traversal. This is reverse-mode automatic differentiation.

**Detail:**
- Zero all `adjoint` slots; set the output node's `adjoint = 1.0`.
- Iterate nodes in **reverse topological order**. For each node, read its `adjoint` (call it `ḡ`), and push contributions into its inputs' `adjoint` slots using the op's local derivative from §4.3 — **accumulate with `+=`** (critical for shared nodes; this is the sum-over-paths chain rule).
- After the pass, each variable node's `adjoint` holds `∂f/∂that_var`. Return a `HashMap<String, f64>` gradient.
- ⚠️ **The Rust challenge:** you're reading one node's adjoint while writing to another node's adjoint, both inside the same `Vec`. The borrow checker will resist `&mut` aliasing. **Solution:** read the values you need into locals first, then write — or index the `Vec` in a split way. This is *the* borrow-checker lesson of the project; when you hit the error, paste it to AI and ask it to *explain the aliasing rule*, then fix it yourself. Don't reach for `RefCell`.

**Acceptance criteria:**
- [X] `∇f` for `sin(x*y)+x^2` at `(1.5,2.0)` matches finite differences to 1e-6 (wait for TICKET-104 to automate this; hand-check one value now).
- [X] A variable appearing in two terms correctly **sums** both path contributions.
- [X] One forward + one backward pass yields the full gradient (no per-input re-run).
- [X] You can explain why the borrow-checker error you hit here was correct.

🦀 **Rust concepts introduced (Stage E, hard mode):** `&mut` aliasing and how to satisfy the borrow checker when mutating a `Vec` you're also reading (read-into-locals-then-write, or `split_at_mut`); reverse iteration (`.iter().rev()` / reverse index loop); the discipline of accumulation. This is the ticket that makes you *actually understand* Rust ownership.

**Learn/read:** micrograd's `backward()` (the concept, in Python — then translate the *idea*, not the code, into arena-Rust); "Rust split_at_mut" and "Rust mutate vec while iterating" write-ups.

---

#### TICKET-104 — Finite-difference oracle (validation harness) ⭐
**Branch:** `test/104-finite-diff-oracle`

**Description:** Your correctness backbone. Every gradient the engine produces is auto-checked against a numerical finite-difference approximation. This is how real AD engines are validated and it will catch the subtle sign/accumulation bugs that don't crash.

**Detail:**
- Helper `numerical_gradient(f: impl Fn(&HashMap<String,f64>) -> f64, point, h)` using central differences `(f(x+h) − f(x−h)) / 2h` per variable.
- In `tests/finite_diff.rs`, a table of hand-built expressions evaluated at several random points; assert AD gradient vs numerical to < 1e-5.
- Add a deliberately-broken-derivative test that *should* fail, to prove the oracle bites.

**Acceptance criteria:**
- [X] ≥ 8 distinct hand-built expressions pass AD-vs-finite-difference at multiple points.
- [X] Breaking one op's derivative makes the harness fail.

🦀 **Rust concepts introduced (Stages F, G):** closures (`impl Fn(...) -> f64`) to pass functions around; `Result`/`Option` in test helpers; `#[test]`, `assert!`, `cargo test`; `rand` crate (add your first external dependency to `Cargo.toml`); generics via `impl Trait` in argument position.

**Learn/read:** central vs forward differences; step-size tradeoff; Rust closures (*Rust Book* ch. 13); adding a crate dependency.

---

### PHASE 2 — Extend the math

---

#### TICKET-200 — Full op set + local-derivative table + error handling
**Branch:** `feat/200-full-ops`

**Description:** Complete every op in the §4.3 table (forward + local derivative), and introduce proper Rust error handling to replace panics.

**Detail:**
- Implement forward + backward for all ops in §4.3; keep each op's forward and derivative **adjacent in code** so they can't drift.
- Define `enum EngineError { UnknownVariable(String), DivByZero, DomainError(String), ... }`; make `forward`/`backward` return `Result<_, EngineError>`; propagate with `?`.
- Guard `ln(x≤0)`, `div` by 0 with clear errors.

**Acceptance criteria:**
- [X] Every §4.3 op has forward + derivative, each finite-difference validated.
- [X] Domain errors return a descriptive `EngineError`, not a panic.

🦀 **Rust concepts introduced (Stage F, full):** custom error `enum`; `Result<T, E>` as return type; the `?` operator for propagation; `impl std::error::Error`/`Display` for your error; when to `panic!` (bugs) vs return `Err` (expected failures).

**Learn/read:** *Rust Book* ch. 9 (error handling); the `?` operator; `thiserror` crate (optional, ergonomic error derives).

---

#### TICKET-201 — Jacobian (multi-output)
**Branch:** `feat/201-jacobian`

**Description:** Extend from a scalar output to a vector function `f: ℝⁿ → ℝᵐ`; assemble the m×n Jacobian by running the backward pass once per output.

**Detail:**
- Support multiple output node indices sharing one graph.
- `fn jacobian(&mut self, outputs: &[usize], vars: &[String]) -> Vec<Vec<f64>>`: for each output, zero adjoints, seed that output to 1, run backward, collect the row.
- Return a dense `Vec<Vec<f64>>` (m rows × n cols).

**Acceptance criteria:**
- [ ] Jacobian of a known 2→2 map (e.g. polar→cartesian) matches the analytic Jacobian to 1e-6.
- [ ] Each row independently finite-difference validated.

🦀 **Rust concepts introduced:** `Vec<Vec<f64>>` nested collections; slices (`&[usize]`, `&[String]`) as function args; iterating with indices to fill a matrix.

**Learn/read:** Jacobian definition (§4.4); why reverse mode = one row per pass.

---

#### TICKET-205 — Trace emission (frontend contract, serde)
**Branch:** `feat/205-trace-serde`

**Description:** Emit the ordered forward/backward trace (§7.3) as a serializable Rust struct, so the visualizer can animate AD step-by-step later. Build now to avoid a painful retrofit.

**Detail:**
- Structs mirroring §7.3 (`Trace`, `TraceNode`, `ForwardStep`, `BackwardStep`), deriving `serde::Serialize`.
- `fn trace(&mut self, inputs) -> Trace` producing forward steps in topo order and backward steps in reverse-topo order.
- A golden-file test pinning the trace JSON for `sin(x*y)+x^2` at a fixed point.

**Acceptance criteria:**
- [ ] `serde_json::to_string(&trace)` produces JSON matching §7.3.
- [ ] `forward.len() == node_count`; `backward` is exactly the reverse order.
- [ ] Golden-file test passes.

🦀 **Rust concepts introduced (Stage K):** `serde` + `serde_json`; `#[derive(Serialize)]`; adding and using ecosystem crates; struct-to-JSON mapping; golden-file testing.

**Learn/read:** serde.rs quickstart; `#[serde(rename)]` for field naming.

---

### PHASE 3 — Compiler front end (now that Rust + graph are solid)

> You now know Rust and have a working graph. The lexer/parser is "just" a way to build that graph from text — and parsing produces a *tree* (single ownership), which is Rust's *easy* case. This phase should feel much smoother than Phase 1.

---

#### TICKET-300 — Lexer
**Branch:** `feat/300-lexer`

**Description:** Turn source text like `sin(x*y) + x^2` into a token stream.

**Detail:**
- `enum Token { Ident(String), Number(f64), Plus, Minus, Star, Slash, Caret, LParen, RParen, Comma, Eof }`.
- `struct Lexer` over the input `char`s (a `Peekable<Chars>` is handy); `next_token(&mut self) -> Result<Token, EngineError>`.
- Handle floats, whitespace-skipping, and a clear error with position on an unexpected char.

**Acceptance criteria:**
- [ ] Lexing `sin(x*y) + x^2` yields the expected 11 tokens + `Eof`.
- [ ] Tests cover multi-digit floats, all operators, nested parens, one error case.

🦀 **Rust concepts introduced:** `char` handling; `Peekable` iterators (`.peek()`, `.next()`); `String` building; `Result`-returning iteration; the `chars()` iterator. Lexing is beginner-friendly Rust — a good confidence rebuild after Phase 1.

**Learn/read:** *Crafting Interpreters* "Scanning"; Rust `Peekable`.

---

#### TICKET-301 — Pratt parser → AST
**Branch:** `feat/301-parser`

**Description:** Parse the token stream into an AST with correct precedence and associativity, using Pratt (top-down operator-precedence) parsing.

**Detail:**
- `enum Expr { Num(f64), Var(String), Unary { op, child: Box<Expr> }, Binary { op, left: Box<Expr>, right: Box<Expr> }, Call { fn_name: String, arg: Box<Expr> } }` — **`Box` is required** for the recursion (a Rust enum can't contain itself by value).
- Pratt core: `parse_expr(&mut self, min_bp: u8) -> Result<Expr, EngineError>` with binding-power tables. Precedence low→high: `+ -` < `* /` < unary `-` < `^` (right-assoc).
- Descriptive syntax errors (unexpected token, unclosed paren).

**Acceptance criteria:**
- [ ] `x^2^3` parses right-associatively as `x^(2^3)`.
- [ ] `-x^2` parses as `-(x^2)`.
- [ ] `a + b*c` groups the multiply first.
- [ ] ≥ 3 error cases return descriptive errors, not panics.

🦀 **Rust concepts introduced (Stages H, I):** `Box<T>` for recursive enums (**the key lesson here** — why the AST *needs* `Box`); recursive functions over an owned tree; single-ownership on a tree (contrast with the arena you needed for the graph — a great thing to articulate: *tree = ownership easy, graph = ownership hard, hence arena*); more `match`.

**Learn/read:** Bob Nystrom, *"Pratt Parsers"*; "Rust recursive enum Box"; the matklad *"Simple but Powerful Pratt Parsing"* post (Rust-specific, excellent).

---

#### TICKET-302 — AST → graph lowering (hash-consing)
**Branch:** `feat/302-lowering`

**Description:** Walk the AST and build the arena graph from TICKET-100/101, deduplicating identical subexpressions (hash-consing) so shared work is a shared node.

**Detail:**
- Recursive `lower(&mut self, expr: &Expr, graph: &mut Graph) -> usize` returning the node index for each subexpression.
- **Hash-consing:** a `HashMap<NodeKey, usize>` from `(op, inputs, attr)` → existing index; reuse instead of duplicating. This is what makes CSE mostly free and connects `x` appearing twice to one shared node.
- Map variable names to their (deduped) var-node indices.

**Acceptance criteria:**
- [ ] `x*y + x*y` lowers to a graph where `x*y` is a single shared node (assert node count).
- [ ] Parsing + lowering `sin(x*y)+x^2` produces a graph whose forward/backward match the hand-built version from Phase 1 exactly.

🦀 **Rust concepts introduced (Stage J):** `HashMap` with a custom key (deriving `Hash`, `Eq`, `PartialEq` on a key struct/enum); recursion returning indices; bridging the tree (AST) and the arena (graph) — a concrete lesson in *why* the two data structures use different ownership strategies.

**Learn/read:** hash-consing / structural sharing; `#[derive(Hash, Eq, PartialEq)]`; using a struct as a `HashMap` key.

---

### PHASE 4 — Optimization passes

---

#### TICKET-400 — Constant folding
**Branch:** `feat/400-const-folding`

**Description:** Evaluate all-constant subgraphs at compile time, replacing them with a single constant node.

**Detail:** Single topological pass; if all a node's inputs are `Const`, compute and replace with `Const`. Re-run to fixpoint or fold in one sweep (inputs precede consumers).

**Acceptance criteria:**
- [ ] `x + 2*3` folds `2*3 → 6`, leaving `x + 6` (assert node count drop).
- [ ] Property test: random points give identical results pre/post fold.

🦀 **Rust concepts introduced:** in-place `Vec` mutation/rewriting; matching on `OpType` + input kinds; `#[cfg(test)]` property-style loops.

**Learn/read:** constant folding as a classic compiler pass; fixpoint iteration.

---

#### TICKET-401 — Common-subexpression elimination
**Branch:** `feat/401-cse`

**Description:** Merge structurally identical nodes into one. Mostly handled by hash-consing at lowering, but implement as a distinct, visible pass over an arbitrary graph so it's demonstrable and benchmarkable.

**Detail:** Key each node by `(op, canonicalized inputs, attr)`; redirect duplicates to a canonical index; drop orphans. Canonicalize commutative operands (`a*b` == `b*a`).

**Acceptance criteria:**
- [ ] A graph with duplicated subexpressions collapses to minimal shared form (assert node count).
- [ ] Result-preserving property test passes.

🦀 **Rust concepts introduced:** `HashMap`-based value numbering; canonical ordering (`.sort()` on inputs); index remapping across the arena.

**Learn/read:** value numbering / CSE; commutative canonicalization.

---

#### TICKET-402 — Dead-node elimination + node-count benchmark
**Branch:** `feat/402-deadcode-bench`

**Description:** Remove nodes unreachable from the output(s); record before/after node counts as a benchmark artifact.

**Detail:** Reverse-reachability mark from output(s); drop unmarked; renumber indices. `bench/` script runs a suite through the full pass pipeline, writes raw-vs-optimized node counts to `bench/results/`.

**Acceptance criteria:**
- [ ] Unreachable nodes removed; output value unchanged.
- [ ] Benchmark artifact committed; README cites a reduction figure.

🦀 **Rust concepts introduced:** graph traversal (reachability) over the arena; `Vec<bool>` mark sets; index renumbering with a remap table.

**Learn/read:** reachability-based DCE; presenting optimization wins quantitatively.

---

### PHASE 5 — Solvers

---

#### TICKET-500 — LU linear solver
**Branch:** `feat/500-lu-solver`

**Description:** Dense `A x = b` solver via LU decomposition with partial pivoting — the workhorse inside Newton.

**Detail:** `lu_decompose(a) -> (l, u, piv)`; `lu_solve(...) -> x`. Partial pivoting for stability; detect singular matrices (return `Result`).

**Acceptance criteria:**
- [ ] Solves random well-conditioned systems to 1e-9 (`A x ≈ b`).
- [ ] Reports singularity gracefully.

🦀 **Rust concepts introduced:** 2D data as `Vec<Vec<f64>>` (or a flat `Vec` + stride — mention the perf tradeoff); nested-loop numerics; `Result` for singular cases. Reuses your matmul-project linear-algebra intuition.

**Learn/read:** LU decomposition; partial pivoting; condition number intuition.

---

#### TICKET-501 — Newton's method
**Branch:** `feat/501-newton`

**Description:** Solve `f(x)=0` for a vector using engine Jacobians + the LU solver.

**Detail:** Loop: eval `f(x)`, build `J`, solve `J Δx = −f(x)`, update, stop on `‖f(x)‖ < tol` or max iters. Emit per-iteration `(x, ‖f(x)‖)` for streaming/plots.

**Acceptance criteria:**
- [ ] Converges on a known system (e.g. circle ∩ line) to the analytic root.
- [ ] Records the quadratic-convergence tail (nice README figure).

🦀 **Rust concepts introduced:** structuring an iterative algorithm; returning a result struct with history (`Vec<Iteration>`); norms over a `Vec`.

**Learn/read:** Newton for systems; convergence conditions; damping/line search basics.

---

#### TICKET-502 — IK solver (Jacobian-transpose → damped least squares)
**Branch:** `feat/502-ik-solver`

**Description:** Drive a 2D jointed arm's end-effector to a target using engine-produced Jacobians. Flagship demo.

**Detail:** Forward kinematics `p(θ)` as an engine expression (cumulative-angle sums of cos/sin). Start with Jacobian-transpose `Δθ = α Jᵀ(t−p)`; upgrade to DLS `Δθ = Jᵀ(JJᵀ + λ²I)⁻¹(t−p)` (LU on the small system). Emit per-iteration joint angles + tip position.

**Acceptance criteria:**
- [ ] 3-link arm reaches a reachable target within tolerance.
- [ ] DLS visibly smoother than transpose near singularities.

🦀 **Rust concepts introduced:** composing the engine's own API to build a parametric function; matrix-vector ops on `Vec`s; the transpose-vs-DLS tradeoff in code.

**Learn/read:** Buss IK paper; planar forward kinematics; damping factor λ.

---

### PHASE 6 — Rust engine as a service

---

#### TICKET-600 — Expose engine over HTTP/JSON (axum + serde)
**Branch:** `feat/600-engine-service`

**Description:** Wrap the engine in a small Rust web service so the Go layer (Tier 2) can call it. This is your first async Rust — kept minimal.

**Detail:**
- Use `axum` + `tokio`. Endpoints: `POST /functions` (parse+lower+optimize, return an id + variables), `POST /eval`, `POST /grad`, `POST /jacobian`, `POST /trace`, `POST /solve` (returns iterations, or streams — streaming can wait for the Go WS layer).
- Requests/responses are serde structs. Keep an in-memory map of compiled functions (id → Graph) behind a `Mutex` or `tokio` state.
- Engine logic stays sync/pure; only the thin HTTP layer is async.

**Acceptance criteria:**
- [ ] `curl` a function submission, then eval and grad round-trip over HTTP.
- [ ] Engine errors map to 4xx JSON, not 500s/panics.

🦀 **Rust concepts introduced (Stage L):** `async`/`await` basics; `tokio` runtime; `axum` handlers, extractors, `Json<T>`; shared state (`Arc<Mutex<...>>` or axum `State`); the sync-core / async-shell split. **Keep this minimal** — you want a working endpoint, not deep async mastery.

**Learn/read:** axum "hello world" + JSON example; `tokio` basics; `Arc<Mutex<T>>` for shared state. Don't over-invest in async theory here.

---

### PHASE 7 — Go service layer (BFF) · Tier 2

> Back in familiar Go territory. This layer calls the Rust engine over HTTP, adds WebSocket streaming to the browser, and (Tier 3) Postgres. No Rust here.

---

#### TICKET-700 — Go server + engine client + REST
**Branch:** `feat/700-go-bff`

**Description:** A Go service that proxies/orchestrates the Rust engine and serves the browser.

**Detail:** `engineclient` package wrapping the Rust engine's HTTP API (typed Go structs). REST endpoints for submit/eval/grad that call through. Thin handlers; all math stays in Rust.

**Acceptance criteria:**
- [ ] Submit → eval → grad works end-to-end through Go → Rust.
- [ ] Engine errors surface as clean Go HTTP errors.

**Learn/read:** Go `net/http`; JSON client patterns; separating transport from the engine.

---

#### TICKET-701 — WebSocket streaming (trace + solver)
**Branch:** `feat/701-ws-streaming`

**Description:** Stream the AD animation trace and per-iteration solver state to the browser over WebSocket.

**Detail:** `/ws/trace` fetches a trace from the engine and streams graph → forward steps → backward steps. `/ws/solve` starts a solve (direct now; Temporal in Phase 8) and forwards each iteration; client-paced animation preferred.

**Acceptance criteria:**
- [ ] Browser receives ordered trace frames and ordered solver iterations.
- [ ] Cancellation via socket close stops server work (`context.Context`).

**Learn/read:** `gorilla/websocket` or `nhooyr/websocket`; framing a WS protocol; Go context cancellation.

---

#### TICKET-702 — Postgres store · Tier 3
**Branch:** `feat/702-postgres`

**Description:** Minimal persistence for saved functions and solver-run history. First thing to cut if short on time.

**Detail:** Tables `functions`, `solver_jobs`, `solver_runs`. Access via `sqlc` (typed SQL) or `pgx` raw. **No ORM.**

**Acceptance criteria:**
- [ ] Migrations apply on a fresh container; save/read round-trips.
- [ ] Store has no engine/HTTP imports.

**Learn/read:** `sqlc` quickstart; `pgxpool`.

---

### PHASE 8 — Durability (Temporal, Go) · Tier 2

---

#### TICKET-800 — Temporal worker skeleton
**Branch:** `feat/800-temporal-worker`

**Description:** Stand up a Temporal worker registering a trivial workflow+activity, proving connectivity.

**Detail:** `worker/main.go` connects to the dev server, registers on task queue `solver-tq`; a ping workflow validates the loop end-to-end.

**Acceptance criteria:**
- [ ] Ping workflow completes and shows in the Temporal UI.

**Learn/read:** Temporal Go SDK setup; Worker/TaskQueue concepts; the dev server UI.

---

#### TICKET-801 — Solve workflow + crash/resume demo
**Branch:** `feat/801-solve-workflow`

**Description:** Model the iterative solver as a durable workflow — each iteration is a checkpointed step — then demo the money shot: kill the worker mid-solve, watch it resume at iteration N.

**Detail:** `SolveWorkflow` loops calling `StepActivity` (which calls the Rust engine for eval + Jacobian + step), appending to workflow state, checking convergence. Workflow code stays **deterministic** (no direct IO/time/random — those live in activities). Wire progress into `/ws/solve`. Document a repeatable `docker kill` mid-solve → restart → continuation.

**Acceptance criteria:**
- [ ] A solve runs to convergence entirely through the workflow.
- [ ] Repeatable crash-resume visibly continues rather than restarting.
- [ ] Passes the Temporal replay test.

**Learn/read:** Temporal determinism rules; activities vs workflow code; the test framework + replay testing; activity idempotency.

---

### PHASE 9 — Frontend · Tier 2/3

---

#### TICKET-900 — App shell + REST wiring
**Branch:** `feat/900-web-shell`

**Description:** React app: enter a function, submit, eval/gradient at a point via REST.

**Detail:** `FunctionInput.tsx`; typed `api/rest.ts`; `types/trace.ts` mirrors §7.3.

**Acceptance criteria:**
- [ ] Submit → evaluate → see gradient in the UI; errors readable.

**Learn/read:** Vite + React + TS basics; TS discriminated unions for node/op types.

---

#### TICKET-901 — Graph visualizer (animated AD)
**Branch:** `feat/901-graph-viz`

**Description:** The standout. Lay out the DAG (dagre/elkjs, layered "IR" look; draw with D3/Canvas) and animate: light up node **values** in forward order, then **adjoints** flowing backward, accumulating the gradient. **Timebox to 3 days** — if the fancy animation fights you, ship a static-graph-with-highlighted-nodes version and move on.

**Detail:** Consume `/ws/trace`; `lib/animate.ts` steps through forward then backward with play/pause/step. Highlight the active node/edge; show each node's value and adjoint.

**Acceptance criteria:**
- [ ] Forward fills values in topo order; backward propagates adjoints in reverse, ending with correct `∂f/∂x`, `∂f/∂y`.
- [ ] Shared node visibly receives both path contributions.
- [ ] Play/pause/step works.

**Learn/read:** dagre/elkjs layout; D3 selections + transitions; driving animation from an ordered event list.

---

#### TICKET-902 — IK arm canvas · Tier 3
**Branch:** `feat/902-ik-canvas`

**Description:** Canvas arm that reaches toward a clicked target, driven live by the solver stream.

**Acceptance criteria:**
- [ ] Clicking a reachable point animates the arm converging; unreachable targets stretch without breaking.

**Learn/read:** Canvas 2D; mapping the iteration stream to frames.

---

#### TICKET-903 — Convergence plot · Tier 3
**Branch:** `feat/903-convergence-plot`

**Description:** Live line chart of solver error vs iteration, streamed in real time (log-scale y optional to show Newton's quadratic tail).

**Acceptance criteria:**
- [ ] Plot updates live; Newton runs show fast late convergence.

**Learn/read:** a lightweight D3 line chart; streaming data into a chart.

---

### PHASE 10 — Benchmarks & README

---

#### TICKET-1000 — Reverse-vs-forward benchmark
**Branch:** `bench/1000-reverse-vs-forward`

**Description:** Empirically demonstrate the headline: reverse mode gets the full gradient in ~one pass regardless of input count; forward/finite-diff scale linearly with input count.

**Detail:** A Rust bench (`criterion` crate, or manual timing) over functions with n = 2,4,8,…,256 inputs, measuring reverse-mode vs forward/finite-difference gradient cost. Write CSV + a plot to `bench/results/`.

**Acceptance criteria:**
- [ ] Results show reverse ≈ flat vs forward/finite-diff ≈ linear in n.
- [ ] README cites the measured asymmetry with numbers.

**Learn/read:** `criterion` benchmarking crate; presenting complexity results honestly.

---

#### TICKET-1001 — README: architecture, results, derivation
**Branch:** `docs/1001-readme`

**Description:** The artifact recruiters read. Architecture diagram, quickstart, both benchmarks, and the one derivation.

**Detail:** Sections: what it is; architecture (diagram); quickstart; the two benchmarks; a short **derivation** of why one reverse pass yields the full gradient (chain rule as sum-over-paths + cost argument). Embed a GIF of the AD animation + IK arm. Note prominently that the engine is written in Rust.

**Acceptance criteria:**
- [ ] A stranger can clone and reach a working demo from the README.
- [ ] The derivation is correct and legible.
- [ ] GIF/screenshots embedded.

**Learn/read:** leading a README with the interesting result; Mermaid diagrams.

---

## 11. Testing & Benchmarks

### 11.1 Testing strategy

**Finite differences are your correctness oracle** (TICKET-104). AD bugs are subtle — a wrong sign or a missing accumulation on a shared node gives wrong numbers without crashing. Numerical gradients catch them automatically. Get this harness green early; every later ticket rides on it.

| Layer | What to test | How |
|---|---|---|
| Arena/graph | structure, shared nodes | assert node counts, index reuse |
| Forward eval | values | vs hand/Python constants |
| **Backward pass** | **gradients** | **vs central finite differences (< 1e-5)** — the backbone |
| Per-op derivatives | local partials | direct unit tests from §4.3 |
| Jacobian | rows/matrix | vs analytic Jacobian |
| Lexer/parser | precedence, assoc, errors | assert on AST shape |
| Lowering | dedup/shared nodes | node counts; parity with hand-built graph |
| Optimizer | value preservation | property test: random points match pre/post |
| LU | `A x = b` | residual `‖Ax−b‖ < 1e-9` |
| Newton/IK | convergence | assert root / tip within tolerance |
| Trace | ordering + schema | golden file |
| Workflow (Tier 2) | determinism | Temporal replay test |
| Service (Tier 2) | round-trips, errors | HTTP client tests |

**Property testing** fits the optimizer perfectly: for a random expression and point, value must be identical before and after each pass. Rust's `proptest` crate is a good fit (and a nice extra Rust thing to learn).

### 11.2 Benchmarks (the two numbers that sell the project)

1. **Reverse vs. forward/finite-difference cost as input dimension grows** (TICKET-1000) — the headline; reverse ≈ flat, others ≈ linear. This *is* why backprop scales; measuring it is your strongest single artifact.
2. **Graph node count before/after optimization passes** (TICKET-402) — the "compiler engineer" signal.

Optional third: **Newton quadratic-convergence tail** — log-scale error-vs-iteration.

### 11.3 Definition of done

**Tier 1 (the standalone Rust engine):**
- [ ] Hand-built and parsed graphs both work; forward + backward validated against finite differences across a suite.
- [ ] Jacobian + one solver working.
- [ ] Reverse-vs-forward benchmark + README with the derivation.

**Tier 2 (the full system):**
- [ ] `docker compose up` → Rust engine + Go BFF + Temporal live.
- [ ] Animated graph visualizer showing forward values then backward adjoints.
- [ ] A solver run survives a worker crash and resumes mid-iteration.

---

## 12. Resume Framing

A line once you've shipped (trim clauses to match what you actually built):

> *Built a differentiable expression compiler in **Rust** — hand-written lexer + Pratt parser, arena-based DAG intermediate representation, reverse-mode automatic differentiation, and constant-folding/CSE optimization passes — exposing exact gradient and Jacobian computation. Wrapped it as a service driving durable, resumable Newton and inverse-kinematics solvers orchestrated in Go with Temporal, with a real-time TypeScript/React + D3 visualizer animating the backward pass.*

**Tier-1-only version (accurate if you stop after the engine):**

> *Built a differentiable expression compiler in **Rust**: lexer, Pratt parser, arena-based computation-graph IR, reverse-mode automatic differentiation, and optimization passes (constant folding, CSE), computing exact gradients and Jacobians. Validated against finite differences; benchmarked reverse-mode's constant-cost gradient vs. forward mode's linear scaling.*

**Story vs. last project:** last time was distributed compute + SIMD + perf (Go/C++/gRPC/Redis/CMake). This is **compilers/PL + automatic differentiation + Rust + (Tier 2) durable orchestration + full-stack**. New languages/tech: Rust, Temporal, TypeScript/React, Postgres, WebSocket. The only carryover is Go (now orchestration) and Docker.

**Interview talking points banked:**
- Learning Rust on a graph-heavy project, and *why the arena/index pattern* beats `Rc<RefCell>` for a multi-parent DAG (a sophisticated Rust-ownership answer most beginners can't give).
- Why AD ≠ symbolic ≠ numerical differentiation, and the reverse-mode cost asymmetry (with a benchmark).
- Tree vs. graph ownership: the AST uses `Box` (single ownership), the IR uses an arena (shared) — and *why*.
- Constant folding / CSE / dead-code elimination as real passes with measured wins.
- (Tier 2) Modeling an iterative solver as a durable Temporal workflow; the Rust-engine / Go-orchestration split at a real compute-vs-serving seam.

---

*Old (Go-engine) roadmap preserved as `gradient-engine-roadmap.BACKUP.md` if you want to compare.*
