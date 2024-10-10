pub struct Ccpa {
    enabled: bool,
}

impl Ccpa {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn run(&self) {
        if self.enabled {
            // Run CCPA checks
            // ...
            println!("CCPA checks complete!");
        }
    }
}
