use elliptic::Elliptic;
use reqwest::Client;

#[test]
fn test_elliptic_run() {
    let api_key = "ELLiptic_API_KEY";
    let api_secret = "ELLiptic_API_SECRET";
    let client = Client::new();
    let elliptic = Elliptic::new(api_key.to_string(), api_secret.to_string(), client);
    elliptic.run().await;
    assert!(true); // Add more assertions here
}
