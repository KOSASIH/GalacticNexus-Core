use compliance::Compliance;
use config::Config;

#[test]
fn test_compliance_run() {
    let config = Config::new();
    let compliance = Compliance::new(config);
    compliance.run();
    assert!(true); // Add more assertions here
}
