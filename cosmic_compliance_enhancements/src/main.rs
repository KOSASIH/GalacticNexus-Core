use std::env;
use config::Config;
use compliance::Compliance;

fn main() {
    let config = Config::new();
    let compliance = Compliance::new(config);
    compliance.run();
}
