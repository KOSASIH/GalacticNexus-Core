pub struct Aml {
    threshold: u64,
}

impl Aml {
    pub fn new(threshold: u64) -> Self {
        Self { threshold }
    }

    pub fn run(&self) {
        // Run AML checks
        // ...
        println!("AML checks complete!");
    }
}
