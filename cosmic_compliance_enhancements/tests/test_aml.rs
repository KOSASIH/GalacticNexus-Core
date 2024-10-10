use aml::Aml;

#[test]
fn test_aml_run() {
    let aml = Aml::new(1000);
    aml.run();
    assert!(true); // Add more assertions here
}
