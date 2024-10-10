use chainalysis::Chainalysis;
use reqwest::Client;

#[test]
fn test_chainalysis_run() {
    let api_key = "CHAINALYSIS_API_KEY";
    let api_secret = "CHAINALYSIS_API_SECRET";
    let client = Client::new();
    let chainalysis = Chainalysis::new(api_key.to_string(), api_secret.to_string(), client);
    chainalysis.run().await;
    assert!(true); // Add more assertions here
}
