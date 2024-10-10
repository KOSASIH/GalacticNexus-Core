use ccpa::Ccpa;

#[test]
fn test_ccpa_run() {
    let ccpa = Ccpa::new(true);
    ccpa.run();
    assert!(true); // Add more assertions here
}
