use std::collections::HashMap;
use aml_kyc::{Aml, Kyc};
use regulatory_frameworks::{Gdpr, Ccpa};

pub struct Compliance {
    config: Config,
    aml: Aml,
    kyc: Kyc,
    gdpr: Gdpr,
    ccpa: Ccpa,
}

impl Compliance {
    pub fn new(config: Config) -> Self {
        let aml = Aml::new(config.aml_threshold);
        let kyc = Kyc::new(config.kyc_threshold);
        let gdpr = Gdpr::new(config.gdpr_enabled);
        let ccpa = Ccpa::new(config.ccpa_enabled);
        Self {
            config,
            aml,
            kyc,
            gdpr,
            ccpa,
        }
    }

    pub fn run(&self) {
        // Run AML checks
        self.aml.run();

        // Run KYC checks
        self.kyc.run();

        // Run GDPR checks
        if self.gdpr.enabled {
            self.gdpr.run();
        }

        // Run CCPA checks
        if self.ccpa.enabled {
            self.ccpa.run();
        }
    }
}
