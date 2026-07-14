//! Trace emission: the engine → frontend contract.
//!
//! Serializes a completed forward/backward pass into the JSON the visualizer
//! animates: a shared node list plus two ordered arrays, `forward` in
//! topological order and `backward` in reverse-topological order. The structs
//! here mirror that contract exactly, so `serde_json` produces the wire shape
//! directly with no post-processing.

use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use serde::Serialize;
use std::collections::HashMap;

/// The whole trace: the graph plus the two ordered step arrays over it.
#[derive(Serialize)]
pub struct Trace {
    pub graph: TraceGraph,
    pub forward: Vec<ForwardStep>,
    pub backward: Vec<BackwardStep>,
}

/// The static graph structure: every node, and the id of the output node.
#[derive(Serialize)]
pub struct TraceGraph {
    pub nodes: Vec<TraceNode>,
    pub output: usize,
}

/// One node in the graph. `op` is a flat lowercase tag; the payload an `OpType`
/// carries is split out into `label` (variable name) or `attr` (op parameter),
/// so the frontend never has to unpack an enum.
#[derive(Serialize)]
pub struct TraceNode {
    pub id: usize,
    pub op: String,

    // Only some ops carry these, so `None` is omitted from the JSON entirely
    // rather than emitted as `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attr: Option<Attr>,
}

/// A scalar op parameter, nested as `{ "k": ... }` (e.g. the exponent of `pow`).
#[derive(Serialize)]
pub struct Attr {
    k: f64,
}

/// One forward step: a node's computed value, emitted in topological order.
#[derive(Serialize)]
pub struct ForwardStep {
    pub id: usize,
    pub value: f64,
}

/// One backward step: a node's adjoint, emitted in reverse-topological order.
#[derive(Serialize)]
pub struct BackwardStep {
    pub id: usize,
    pub adjoint: f64,
}

impl Graph {
    /// Run both AD passes and package them as a serializable [`Trace`].
    ///
    /// `output` is the node index to differentiate. The caller owns "which node
    /// is the root," matching [`Graph::backward`] and [`Graph::jacobian`]. It
    /// seeds the backward pass and is recorded as [`TraceGraph::output`].
    ///
    /// `forward` is emitted in topological (index) order; `backward` is the full
    /// node list in reverse. Nodes that are not ancestors of `output` simply
    /// carry adjoint `0.0`, since the backward pass never reached them.
    ///
    /// Fails only if [`Graph::forward`] or [`Graph::backward`] do (for example an
    /// unknown variable or a domain error).
    pub fn trace(
        &mut self,
        inputs: &HashMap<String, f64>,
        output: usize,
    ) -> Result<Trace, EngineError> {
        // Populate every node's `value` before we snapshot the forward steps.
        self.forward(inputs)?;
        let mut graph_nodes: Vec<TraceNode> = Vec::new();
        let mut forward_nodes: Vec<ForwardStep> = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            // Translate the rich `OpType` enum into the flat wire form: a
            // lowercase tag, with any payload lifted out into `label` or `attr`.
            let (op, label, attr) = match &node.op {
                OpType::Var(name) => ("var", Some(name.clone()), None),
                OpType::Pow(n) => ("pow", None, Some(Attr { k: *n })),
                OpType::Const(c) => ("const", None, Some(Attr { k: *c })),
                OpType::Add => ("add", None, None),
                OpType::Sub => ("sub", None, None),
                OpType::Neg => ("neg", None, None),
                OpType::Div => ("div", None, None),
                OpType::Mul => ("mul", None, None),
                OpType::Sin => ("sin", None, None),
                OpType::Cos => ("cos", None, None),
                OpType::Exp => ("exp", None, None),
                OpType::Ln => ("ln", None, None),
            };

            let graph_node: TraceNode = TraceNode {
                id: i,
                op: op.to_string(),
                label,
                // Leaves (vars/consts) have no inputs, so omit the field for them.
                inputs: if node.inputs.is_empty() {
                    None
                } else {
                    Some(node.inputs.clone())
                },
                attr,
            };

            graph_nodes.push(graph_node);
            forward_nodes.push(ForwardStep {
                id: i,
                value: node.value,
            });
        }

        let trace_graph: TraceGraph = TraceGraph {
            nodes: graph_nodes,
            output,
        };

        // Populate every node's `adjoint` by seeding `output`, then snapshot the
        // whole arena in reverse. The bound is the full node list, NOT `0..=output`:
        // reverse-topological order covers every node wherever the root sits, and
        // nodes above `output` are correctly left at adjoint `0.0`.
        self.backward(output)?;
        let mut backward_nodes: Vec<BackwardStep> = Vec::new();
        for i in (0..self.nodes.len()).rev() {
            backward_nodes.push(BackwardStep {
                id: i,
                adjoint: self.nodes[i].adjoint,
            });
        }

        Ok(Trace {
            graph: trace_graph,
            forward: forward_nodes,
            backward: backward_nodes,
        })
    }
}
