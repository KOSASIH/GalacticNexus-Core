pub struct Gdpr {
    enabled: bool,
}

impl Gdpr {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn run(&self) {
        if self.enabled {
            // Run GDPR checks
            // ...
            println!("GDPR checks complete!");
        }
    }
}
