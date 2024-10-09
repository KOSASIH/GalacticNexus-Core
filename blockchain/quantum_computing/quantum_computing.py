operation QuantumHelloWorld() : Result {
    using (qubit = Qubit()) {
        H(qubit);
        let result = M(qubit);
        Reset(qubit);
        return result;
    }
}
