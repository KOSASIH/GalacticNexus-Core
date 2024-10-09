# risk_analysis.py

from risk_factors import risk_factors
from data_loader import load_data
from risk_model import train_risk_model

def perform_risk_analysis():
    data = load_data("risk_data.csv")
    model, mse = train_risk_model(data)
    risk_scores = []
    for risk_factor in risk_factors:
        risk_score = model.predict(risk_factor.impact)
        risk_scores.append((risk_factor.name, risk_score))
    return risk_scores

# Perform risk analysis
risk_scores = perform_risk_analysis()
print(risk_scores)
