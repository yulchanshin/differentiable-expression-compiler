//! HTTP API round-trip + error-handling tests (TICKET-600).
//!
//! Drives the router in-process via `oneshot` (no bound socket). One `app()` is
//! built and cloned per request so every call shares the same registry (a clone
//! shares the `Arc` state), letting a test create a function then use its id.

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use engine::api::app;
use serde_json::{Value, json};
use tower::ServiceExt; // for `oneshot`

/// POST `body` as JSON to `uri` on a clone of `router`; return status + parsed body.
async fn post(router: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() < 1e-9
}

#[tokio::test]
async fn submit_eval_grad_round_trip() {
    let router = app();

    // Submit a function.
    let (status, body) = post(&router, "/functions", json!({ "source": "sin(x*y)+x^2" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], "fn-1");
    assert_eq!(body["variables"], json!(["x", "y"]));

    // Evaluate it: sin(1.5*2) + 1.5^2 = sin(3) + 2.25.
    let point = json!({ "id": "fn-1", "inputs": { "x": 1.5, "y": 2.0 } });
    let (status, body) = post(&router, "/eval", point.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(close(body["value"].as_f64().unwrap(), (3.0_f64).sin() + 2.25));

    // Gradient: df/dx = y*cos(xy) + 2x, df/dy = x*cos(xy).
    let (status, body) = post(&router, "/grad", point).await;
    assert_eq!(status, StatusCode::OK);
    let cos3 = (3.0_f64).cos();
    assert!(close(body["gradient"]["x"].as_f64().unwrap(), 2.0 * cos3 + 3.0));
    assert!(close(body["gradient"]["y"].as_f64().unwrap(), 1.5 * cos3));
}

#[tokio::test]
async fn trace_returns_graph_and_steps() {
    let router = app();
    post(&router, "/functions", json!({ "source": "x*y" })).await;

    let (status, body) = post(
        &router,
        "/trace",
        json!({ "id": "fn-1", "inputs": { "x": 2.0, "y": 3.0 } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["graph"]["nodes"].is_array());
    assert!(body["forward"].is_array());
    assert!(body["backward"].is_array());
}

#[tokio::test]
async fn bad_source_is_400_json() {
    let router = app();
    let (status, body) = post(&router, "/functions", json!({ "source": "sin(" })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn unknown_function_id_is_404() {
    let router = app();
    let (status, body) = post(
        &router,
        "/eval",
        json!({ "id": "fn-99", "inputs": { "x": 1.0 } }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn missing_variable_is_400() {
    let router = app();
    post(&router, "/functions", json!({ "source": "x*y" })).await;
    let (status, _) = post(
        &router,
        "/eval",
        json!({ "id": "fn-1", "inputs": { "x": 1.5 } }), // y missing
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
