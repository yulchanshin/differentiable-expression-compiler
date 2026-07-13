use serde::Serialize;

#[derive(Serialize)]
pub struct Trace {
    pub graph: TraceGraph,
    pub forward: Vec<ForwardStep>,
    pub backward: Vec<BackwardStep>,
}

#[derive(Serialize)]
pub struct TraceGraph {
    pub nodes: Vec<TraceNode>,
    pub output: usize,
}

#[derive(Serialize)]
pub struct TraceNode {
    pub id: usize,
    pub op: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attr: Option<Attr>,
}

#[derive(Serialize)]
pub struct Attr {
    k: f64,
}

#[derive(Serialize)]
pub struct ForwardStep {
    pub id: usize,
    pub value: f64,
}

#[derive(Serialize)]
pub struct BackwardStep {
    pub id: usize,
    pub adjoint: f64,
}
