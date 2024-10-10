use gdpr::Gdpr;

#[test]
fn test_gdpr_run() {
    let gdpr = Gdpr::new(true);
    gdpr.run();
    assert!(true); // Add more assertions here
}
