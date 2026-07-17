# Gradient Engine

A differentiable expression compiler written in Rust. It takes a mathematical
function over several variables, for example `f(x, y) = sin(x*y) + x^2`, compiles
it into a computational graph, and then evaluates it and differentiates it
*exactly* using reverse-mode automatic differentiation. One backward pass yields
the full gradient, every partial derivative at once.

The point isn't "take a derivative." It's to show, end to end, a real compiler
pipeline (lexer, Pratt parser, graph IR, execution) with automatic
differentiation as the payload, and to demonstrate the one asymmetry that makes
backpropagation matter: reverse-mode AD computes the entire gradient in a single
pass at the cost of about one function evaluation, regardless of how many inputs
the function has. Finite differences and forward-mode cost one pass per input.

The calculus is the payload. The substance is programming languages and
compilers.

## Status

The Rust engine is the current deliverable and the focus of this README. The
front end and automatic differentiation core are implemented and tested; the
optimizer, solvers, and surrounding service layers are scaffolded and land next
(see [What's next](#whats-next)).

| Stage                                       | State       |
| ------------------------------------------- | ----------- |
| Computational graph (arena DAG)             | Implemented |
| Front end (lexer, Pratt parser, lowering)   | Implemented |
| Forward evaluation                          | Implemented |
| Reverse-mode autodiff (gradient, Jacobian)  | Implemented |
| Trace export for visualization              | Implemented |
| Optimizer (const-fold, CSE, dead-node)      | In progress |
| Solvers (Newton, inverse kinematics)        | Planned     |
| Dense linear algebra (LU with pivoting)     | Planned     |
| Go service layer + browser visualizer       | Planned     |

## The one mental model

A computational graph is a directed acyclic graph. The forward pass is a
topological-order evaluation. The backward pass is a reverse-topological-order
traversal that accumulates adjoints. Everything hangs off that spine.

## Engine architecture

Zoomed in on the Rust engine: a straight compiler pipeline from source text to a
runnable graph, followed by two traversals of that graph. This is the part that
is built.

```
   ENGINE PIPELINE (Rust)

   source text
   "sin(x*y) + x^2"
         |  lex
         v
    token stream --parse--> AST --lower (hash-cons)--> computational graph
                                                        (arena of nodes)
                                                              |
                    +-----------------------------------------+------------------------------------------+
                    |  forward: topological order             |  backward: reverse topological order     |
                    v                                         v
              value  f(x, y)                          gradient  df/dx_i   (one pass, every input)
```

Lowering is *hash-consed*: structurally identical subexpressions collapse to a
single node, so a graph is a DAG with shared operands rather than a tree. This is
what makes the "sum over paths" behavior of the backward pass observable, and it
is the seam the optimizer will later exploit.

## System architecture

Zoomed out to the end-to-end system this repository builds toward. Every layer
lives in this monorepo; most are still planned. The engine stands alone as Tier
1: callable as a library or a thin CLI, it already shows gradients, Jacobians,
and solvers working. The surrounding layers turn a good engine into a full-stack
service. The language boundary sits at a genuine compute-versus-serving seam:
Rust owns exact, fast numerical computation and the compiler, Go owns
orchestration and durability (Temporal is Go-native), and TypeScript owns the
visualization.

```
   FULL SYSTEM (browser + Go service + Rust engine)

```
                    +-----------------------------------------------+
                    |               BROWSER (SPA)                   |
                    |  TypeScript + React + Vite                    |
                    |  graph visualizer (D3 + dagre), IK arm, plot  |   [planned]
                    +-----------------------+-----------------------+
                                            |  WebSocket (JSON frames) + REST
                                            v
        +-------------------------------------------------------------------+
        |                    GO SERVICE LAYER (BFF)                         |   [planned]
        |  REST, WebSocket streaming, Temporal client, Postgres            |
        +----------+----------------------------+-------------------+-------+
                   | HTTP/JSON (or gRPC)         | Temporal          | SQL
                   v                             v                   v
      +--------------------------+   +--------------------+   +--------------+
      |   RUST ENGINE (service)  |   |  TEMPORAL SERVER   |   |  POSTGRES    |
      |   the whole compiler:    |   |  durable, resumable|   |  saved       |
      |   lex -> parse ->        |   |  solver workflows  |   |  functions,  |
      |   graph IR -> optimizer  |   |                    |   |  run history |
      |   -> forward/reverse AD  |   +--------------------+   +--------------+
      |   -> LU / Newton / IK    |        [planned]                [planned]
      +--------------------------+
              [built]
```

Tier 1 stops at the Rust engine. Go, Temporal, and the browser visualizer are
additive layers that do not change the engine's design.

## Quick start

The engine is a library crate. Build and run the test suite from `engine/`:

```bash
cd engine
cargo test          # unit tests + finite-difference validation + golden trace
```

Compile a function, evaluate it, and take its gradient:

```rust
use engine::parse::compile;
use std::collections::HashMap;

// Parse and lower source text into a graph plus the index of its output node.
let (mut graph, output) = compile("sin(x*y) + x^2").unwrap();

let inputs = HashMap::from([
    ("x".to_string(), 1.5),
    ("y".to_string(), 2.0),
]);

let value = graph.forward(&inputs).unwrap();   // evaluate f(1.5, 2.0)
let grad = graph.backward(output).unwrap();     // one reverse-mode pass

println!("f     = {value}");
println!("df/dx = {}", grad["x"]);   // y*cos(x*y) + 2x
println!("df/dy = {}", grad["y"]);   // x*cos(x*y)
```

`backward` requires a preceding `forward` on the same inputs, because the
derivative rules read the node values that the forward pass fills in.

## How it works

### 1. The computational graph (an arena)

Every node lives in one `Vec<Node>` that owns it. Nodes refer to each other by
`usize` index, not by pointer or `Rc`. This sidesteps the borrow-checker pain of
a multi-parent graph: there is exactly one owner (the arena), and an "edge" is
just an index.

```rust
pub struct Node {
    pub op: OpType,         // what this node computes
    pub inputs: Vec<usize>, // indices of its operand nodes
    pub value: f64,         // filled by the forward pass
    pub adjoint: f64,       // filled by the backward pass: d(output)/d(this)
}

pub enum OpType {
    Var(String), Const(f64),
    Add, Sub, Mul, Div, Neg,
    Pow(f64),                 // exponent is a compile-time constant
    Sin, Cos, Exp, Ln,
}
```

The builder helpers (`graph.mul(a, b)`, `graph.sin(a)`, and so on) always push a
node's inputs before the node itself. That single invariant means plain index
order is a valid topological order: a node's inputs always sit at lower indices.
The forward pass is therefore a forward scan, and the backward pass a reverse
scan, with no separate sort needed.

### 2. The front end (lexer, Pratt parser, lowering)

Source text becomes a graph in three stages, each usable on its own.

**Lexer.** A single pass over the characters, skipping whitespace and classifying
what token starts next. Multi-character runs (identifiers, numbers) are consumed
by helpers; everything else is a one-character token. Bad input becomes a typed
error rather than a panic.

```rust
pub enum Token {
    Ident(String), Number(f64),
    Plus, Minus, Star, Slash, Caret,
    LParen, RParen, Comma, Eof,
}

match c {
    c if c.is_alphabetic()             => Ok(self.read_identifier()),
    c if c.is_ascii_digit() || c == '.' => self.read_number(),
    '+' => { self.chars.next(); Ok(Token::Plus) }
    // ... one arm per operator and delimiter ...
    _   => { self.chars.next(); Err(EngineError::UnexpectedChar(c)) }
}
```

**Parser.** A Pratt (precedence-climbing) parser builds the AST. All precedence
and associativity live in one table of `(left_bp, right_bp)` binding powers:
higher binds tighter, and `^` flips `right < left` to become right-associative.
Adding an operator means editing this table, nothing else.

```rust
fn infix_binding_power(tok: &Token) -> Option<(u8, u8)> {
    match tok {
        Token::Plus | Token::Minus => Some((1, 2)),   // loosest, left-assoc
        Token::Star | Token::Slash => Some((3, 4)),
        Token::Caret               => Some((7, 6)),   // tightest, right-assoc
        _ => None,                                     // not an infix operator
    }
}
```

The core loop seeds a left operand, then folds in any operator that binds at
least as tightly as the caller's `min_bp`, recursing on the right operand with
that operator's `right_bp`:

```rust
pub fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, EngineError> {
    let mut lhs = self.parse_atom()?;               // number, var, call, or ( ... )
    loop {
        let op = self.peek().clone();
        let (left_bp, right_bp) = match infix_binding_power(&op) {
            Some(bp) => bp,
            None => break,                          // not an operator: done
        };
        if left_bp < min_bp { break; }              // binds too loosely: leave it
        self.advance();
        let rhs = self.parse_expr(right_bp)?;       // associativity lives in right_bp
        lhs = Expr::Binary { op, left: Box::new(lhs), right: Box::new(rhs) };
    }
    Ok(lhs)
}
```

It supports `+ - * / ^`, prefix `-`, parenthesized groups, and the calls `sin`,
`cos`, `exp`, `ln`. The `^` exponent must be a numeric literal, which becomes
`Pow(k)`.

**Lowering.** Walk the AST and emit arena nodes, hash-consing as we go. Every arm
funnels through one create-or-reuse step: build a structural key from the op and
its already-lowered input indices, and if that key has been seen, return the
existing node instead of pushing a duplicate.

```rust
fn intern(&mut self, graph: &mut Graph, op: OpType, inputs: Vec<usize>) -> usize {
    let key = NodeKey::new(&op, &inputs);
    if let Some(&idx) = self.memo.get(&key) {
        return idx;                    // structurally identical node already exists
    }
    let idx = graph.push(Node { op, inputs, value: 0.0, adjoint: 0.0 });
    self.memo.insert(key, idx);
    idx
}
```

The key stores each `f64` payload as its `to_bits()` pattern, since `f64` is
neither `Eq` nor `Hash`. Because a node is interned only after its inputs, index
order stays a valid topological order and the root is the last node pushed.

The three stages are wrapped by one entry point:

```rust
pub fn compile(src: &str) -> Result<(Graph, usize), EngineError> {
    lower(&parse(lex(src)?)?)
}
```

### 3. Forward evaluation

The forward pass fills every node's `value` by scanning the arena in index order
and matching on the op, reading each input's already-computed value. Domain
failures are returned as typed errors rather than producing `NaN` or `inf` or
panicking: an unknown variable, division by zero, `ln` of a non-positive number,
or `pow` of a negative base to a fractional exponent each yield an
`EngineError`.

### 4. Reverse-mode automatic differentiation (the payload)

The backward pass *is* the multivariable chain rule over the DAG. For an output
$L$, define the **adjoint** of a node $v$ as

$$\bar{v} \;=\; \frac{\partial L}{\partial v}.$$

The rule that propagates adjoints is the sum-over-paths chain rule: a node $u$
collects a contribution from every consumer $v$ it feeds into,

$$\bar{u} \;=\; \sum_{v \,\in\, \mathrm{consumers}(u)} \bar{v}\,\frac{\partial v}{\partial u},$$

where $\partial v / \partial u$ is the **local derivative** of that op. The
implementation realizes this in three steps:

1. Seed the output's adjoint, since $\partial L / \partial L = 1$.
2. Scan nodes in reverse topological order. At each node, take its accumulated
   adjoint $\bar{v}$ and push $\bar{v}\,(\partial v/\partial u)$ to each input
   $u$.
3. A node with multiple parents *accumulates* with `+=`, which is exactly the
   sum above. This is why shared subexpressions matter and why we accumulate
   rather than assign.

When the scan finishes, each variable node holds its partial derivative. The
local derivative rules are the mathematical contract of the engine:

| Op         | Forward `v` | Adjoint pushed to inputs (incoming adjoint `g`) |
| ---------- | ----------- | ----------------------------------------------- |
| `add(a,b)` | `a + b`     | `a += g`;  `b += g`                             |
| `sub(a,b)` | `a - b`     | `a += g`;  `b += -g`                            |
| `mul(a,b)` | `a * b`     | `a += g*b`;  `b += g*a`                          |
| `div(a,b)` | `a / b`     | `a += g/b`;  `b += -g*a/(b*b)`                   |
| `pow(a,k)` | `a^k`       | `a += g*k*a^(k-1)`                              |
| `sin(a)`   | `sin a`     | `a += g*cos a`                                  |
| `cos(a)`   | `cos a`     | `a += -g*sin a`                                 |
| `exp(a)`   | `exp a`     | `a += g*exp a`                                  |
| `ln(a)`    | `ln a`      | `a += g/a`                                       |
| `neg(a)`   | `-a`        | `a += -g`                                        |

In code, the scan seeds the output and walks backward, and each op's arm reads
the node values and pushes into input adjoints. The multiply rule is the
canonical case:

```rust
self.nodes[output].adjoint = 1.0;

for i in (0..len).rev() {
    let g = self.nodes[i].adjoint;   // this node's accumulated adjoint
    match self.nodes[i].op {
        OpType::Mul => {
            // d/da (a*b) = b,  d/db (a*b) = a
            let (a, b) = (self.nodes[i].inputs[0], self.nodes[i].inputs[1]);
            let (av, bv) = (self.nodes[a].value, self.nodes[b].value);
            self.nodes[a].adjoint += g * bv;
            self.nodes[b].adjoint += g * av;
        }
        // ... one arm per op ...
        _ => {}
    }
}
```

Because `x` in `sin(x*y) + x^2` is a single shared node feeding both `x*y` and
`x^2`, `df/dx` correctly sums the contributions from both terms. That is the
sum-over-paths chain rule falling out of `+=` for free.

### 5. Jacobians

For a vector function $f : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the
$m \times n$ matrix of first partials

$$J \;=\; \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix},
\qquad J_{ij} \;=\; \frac{\partial f_i}{\partial x_j}.$$

Reverse mode produces one *row* per backward pass: seed output $i$'s adjoint to 1
and the rest to 0, and the resulting variable adjoints are row $i$. The engine
assembles the dense Jacobian from one forward evaluation followed by $m$ backward
passes. Downstream solvers consume $J$.

### 6. Trace export

For the eventual visualizer, `trace` runs a forward and backward pass and
serializes the whole thing to JSON: the graph structure, the per-node values in
forward order, and the per-node adjoints in backward order. Two ordered arrays
over a shared node list are all the front end needs to animate the computation
step by step. A golden-file test pins the exact output so the contract is stable.

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
  "forward":  [ { "id": 0, "value": 1.5 } /* ... topological order ... */ ],
  "backward": [ { "id": 5, "adjoint": 1.0 } /* ... reverse-topo order ... */ ]
}
```

## Why reverse mode: the asymmetry

Three ways to get a derivative, and why this engine uses the third:

- **Symbolic** (manipulate formulas into formulas): exact, but the expressions
  blow up in size.
- **Numerical** (finite differences, $f'(x) \approx \frac{f(x+h) - f(x)}{h}$):
  approximate, and costs one function evaluation *per input*.
- **Automatic** (apply the chain rule locally at each primitive op): exact, at
  about one function evaluation's cost for the entire gradient.

Reverse mode is the many-inputs, few-outputs case, which is exactly the shape of
gradient descent and backpropagation. The cost of the full gradient is
independent of the input count. That is the property worth proving with a
benchmark, and the reverse-versus-forward node-count and timing comparison lands
here:

```
Benchmark table to come: reverse-mode gradient cost versus finite differences
as the number of inputs grows.
```

## Validation

Correctness is checked against a finite-difference oracle: for a suite of
expressions evaluated at random points, the analytic gradient from the backward
pass must match the numerical estimate within tolerance. Jacobian rows are
validated the same way, and a golden JSON file pins the visualization trace.

```bash
cd engine
cargo test            # includes the finite-difference harness and golden trace
```

## What's next

These modules are scaffolded in the tree and land in upcoming work. Signatures
and snippets are filled in as each ships.

### Optimizer

A small pass pipeline over the graph, run between differentiations to fight
expression swell:

- **Constant folding:** evaluate all-constant subgraphs at compile time and
  replace them with a single constant.
- **Common-subexpression elimination:** merge structurally identical nodes
  (largely handled at lowering by hash-consing, implemented again as an explicit,
  benchmarkable pass).
- **Dead-node elimination:** drop nodes unreachable from the output and record
  before-and-after node counts as a benchmark artifact.

```rust
// to come
```

### Solvers

Iterative solvers built on the Jacobian.

**Newton's method** for root finding. To solve $f(x) = 0$, repeat until
$\lVert f(x) \rVert$ is small:

$$J(x_k)\,\Delta x = -f(x_k), \qquad x_{k+1} = x_k + \Delta x.$$

Each step evaluates $f$ and its Jacobian $J$, then solves one dense linear
system for the step $\Delta x$.

**Inverse kinematics** for a small planar arm. Given joint angles $\theta$, the
forward-kinematics map $p(\theta)$ gives the end-effector position; reaching a
target $t$ means solving $p(\theta) = t$. This is Newton's method on
$f(\theta) = p(\theta) - t$, whose Jacobian $\partial p / \partial \theta$ is
exactly what reverse-mode AD produces.

Both need a dense linear solve, so an **LU decomposition with partial pivoting**
lands alongside them.

```rust
// to come
```

### Service layer and visualizer

A Go service fronts the engine over the wire, and a TypeScript visualizer
animates the differentiation on the actual graph, plus an inverse-kinematics arm
that reaches toward a clicked target.

## Project structure

```
engine/                  the Rust engine (this is the crate)
  src/
    graph/               node + OpType data model, the arena, topo utilities
    parse/               lexer, AST, Pratt parser, hash-consing lowering
    autodiff/            forward eval, reverse-mode backward pass, Jacobian, trace
    ops/                 shared per-op evaluation and derivative rules
    optimize/            constant folding, CSE, dead-node elimination
    solver/              Newton's method, inverse kinematics
    linalg/              LU decomposition with partial pivoting
    error.rs             the engine's typed error enum
  tests/                 finite-difference validation, golden trace

server/                  Go service layer: REST, WebSocket, Temporal client   [planned]
worker/                  Go Temporal workflows for durable solver runs         [planned]
web/                     TypeScript + React visualizer: graph, IK arm, plot    [planned]
bench/                   reverse-vs-forward cost benchmarks and results        [planned]
```

## License

Not yet licensed.
