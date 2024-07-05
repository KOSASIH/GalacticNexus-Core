import numpy as np
from brain_computer_interface import BCI

class NeuroFinancialInterface:
    def __init__(self):
        self.bci = BCI()

    def read_brain_signals(self) -> np.ndarray:
        # Read brain signals using BCI
        return self.bci.read_signals()

    def process_brain_signals(self, signals: np.ndarray) -> str:
        # Process brain signals to generate trading decisions
        return "Buy" if signals.mean() > 0.5 else "Sell"
