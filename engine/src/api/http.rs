//! Router, shared state, request/response DTOs, and handlers.
//!
//! Four endpoints, all POST JSON: `/functions` compiles source and registers it
//! under an id; `/eval`, `/grad`, `/trace` run the corresponding pass on a
//! registered function at a point. Engine failures become 4xx JSON via
//! [`ApiError`] rather than panicking or 500ing (`/jacobian` and `/solve` are
//! deferred — see the TICKET-600 note in the roadmap).

use crate::autodiff::trace::Trace;
use crate::error::EngineError;
use crate::graph::arena::Graph;
use crate::graph::node::OpType;
use crate::parse::compile;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared registry of compiled functions, keyed by assigned id.
type AppState = Arc<Mutex<Registry>>;

#[derive(Default)]
struct Registry {
    fns: HashMap<String, StoredFn>,
    next: u64, // monotonic id counter, so ids need no randomness/clock
}

struct StoredFn {
    graph: Graph,
    output: usize, // the graph's root node, differentiated by /grad and /trace
}

/// Build the router with a fresh, empty registry.
pub fn app() -> Router {
    let state: AppState = Arc::new(Mutex::new(Registry::default()));
    Router::new()
        .route("/functions", post(create_function))
        .route("/eval", post(eval))
        .route("/grad", post(grad))
        .route("/trace", post(trace))
        .with_state(state)
}

/// Bind `addr` and serve until the process exits.
pub async fn serve(addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app()).await
}

// ---- DTOs -----------------------------------------------------------------

#[derive(Deserialize)]
struct CreateReq {
    source: String,
}

#[derive(Serialize)]
struct CreateResp {
    id: String,
    variables: Vec<String>,
}

/// `/eval`, `/grad`, and `/trace` all take the same shape: a function id plus a
/// point to evaluate at.
#[derive(Deserialize)]
struct PointReq {
    id: String,
    inputs: HashMap<String, f64>,
}

#[derive(Serialize)]
struct EvalResp {
    value: f64,
}

#[derive(Serialize)]
struct GradResp {
    gradient: HashMap<String, f64>,
}

// ---- handlers -------------------------------------------------------------

async fn create_function(
    State(state): State<AppState>,
    Json(req): Json<CreateReq>,
) -> Result<Json<CreateResp>, ApiError> {
    let (graph, output) = compile(&req.source)?;
    let variables = variables(&graph);

    let mut reg = state.lock().unwrap();
    reg.next += 1;
    let id = format!("fn-{}", reg.next);
    reg.fns.insert(id.clone(), StoredFn { graph, output });

    Ok(Json(CreateResp { id, variables }))
}

async fn eval(
    State(state): State<AppState>,
    Json(req): Json<PointReq>,
) -> Result<Json<EvalResp>, ApiError> {
    let mut reg = state.lock().unwrap();
    let f = reg.fns.get_mut(&req.id).ok_or_else(|| ApiError::not_found(&req.id))?;
    let value = f.graph.forward(&req.inputs)?;
    Ok(Json(EvalResp { value }))
}

async fn grad(
    State(state): State<AppState>,
    Json(req): Json<PointReq>,
) -> Result<Json<GradResp>, ApiError> {
    let mut reg = state.lock().unwrap();
    let f = reg.fns.get_mut(&req.id).ok_or_else(|| ApiError::not_found(&req.id))?;
    let output = f.output;
    f.graph.forward(&req.inputs)?; // fills node values the backward pass reads
    let gradient = f.graph.backward(output)?;
    Ok(Json(GradResp { gradient }))
}

async fn trace(
    State(state): State<AppState>,
    Json(req): Json<PointReq>,
) -> Result<Json<Trace>, ApiError> {
    let mut reg = state.lock().unwrap();
    let f = reg.fns.get_mut(&req.id).ok_or_else(|| ApiError::not_found(&req.id))?;
    let output = f.output;
    let trace = f.graph.trace(&req.inputs, output)?;
    Ok(Json(trace))
}

/// The distinct variable names in a graph, sorted so column order is stable.
fn variables(graph: &Graph) -> Vec<String> {
    let mut names: Vec<String> = graph
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            OpType::Var(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

// ---- errors ---------------------------------------------------------------

/// Every expected failure maps to a 4xx with a JSON `{ "error": ... }` body.
enum ApiError {
    BadRequest(String), // a compile/eval failure from the engine
    NotFound(String),   // an unknown function id
}

impl ApiError {
    fn not_found(id: &str) -> Self {
        ApiError::NotFound(format!("no function {id}"))
    }
}

/// Engine errors are client errors here: they come from user source or inputs.
impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError::BadRequest(format!("{e:?}"))
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, m),
        };
        (status, Json(ErrorBody { error: message })).into_response()
    }
}
