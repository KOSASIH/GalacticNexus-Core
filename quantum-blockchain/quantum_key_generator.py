use rand::Rng;
use quantum::{QuantumRegister, QuantumGate};

struct QuantumKeyGenerator {
    qr: QuantumRegister,
    gate: QuantumGate,
}

impl QuantumKeyGenerator {
    fn new() -> Self {
        let qr = QuantumRegister::new(256);
        let gate = QuantumGate::new("H");
        Self { qr, gate }
    }

    fn generate_key(&mut self) -> Vec<u8> {
        let mut key = vec![0u8; 256];
        for i in 0..256 {
            self.qr.apply_gate(self.gate.clone());
            let measurement = self.qr.measure();
            key[i] = measurement as u8;
        }
        key
    }
}
