#[tokio::main]
async fn main() {
    let addr = "127.0.0.1:8080";
    println!("engine service listening on http://{addr}");
    engine::api::serve(addr).await.unwrap();
}
