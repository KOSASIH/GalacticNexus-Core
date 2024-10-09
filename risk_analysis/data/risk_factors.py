# risk_factors.py

class RiskFactor:
    def __init__(self, name, description, impact):
        self.name = name
        self.description = description
        self.impact = impact

# Define risk factors
risk_factors = [
    RiskFactor("Market Volatility", "Risk of market fluctuations affecting investment", 0.5),
    RiskFactor("Regulatory Changes", "Risk of changes in regulations affecting business operations", 0.3),
    RiskFactor("Cybersecurity Threats", "Risk of cyber attacks affecting data security", 0.8),
    # Add more risk factors as needed
]
