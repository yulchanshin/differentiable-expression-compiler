//! HTTP service: a thin axum shell exposing the engine over JSON.
//!
//! The engine core stays sync and pure; each handler locks a shared function
//! registry, runs the (synchronous) pass, and releases before serializing — so
//! no lock is ever held across an `.await`. See [`http::app`] for the router.

pub mod http;

pub use http::{app, serve};
