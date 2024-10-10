use kyc::Kyc;

#[test]
fn test_kyc_run() {
    let kyc = Kyc::new(500);
    kyc.run();
    assert!(true); // Add more assertions here
}
