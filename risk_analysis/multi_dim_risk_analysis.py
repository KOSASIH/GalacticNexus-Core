import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from risk_models.financial_risk_model import FinancialRiskModel
from risk_models.operational_risk_model import OperationalRiskModel
from risk_models.reputational_risk_model import ReputationalRiskModel
from utils.math_utils import calculate_correlation_matrix
from utils.data_utils import load_risk_data, load_risk_factors

class MultiDimRiskAnalysis:
    def __init__(self, risk_data, risk_factors):
        self.risk_data = risk_data
        self.risk_factors = risk_factors
        self.financial_risk_model = FinancialRiskModel()
        self.operational_risk_model = OperationalRiskModel()
        self.reputational_risk_model = ReputationalRiskModel()

    def calculate_risk_scores(self):
        financial_risk_score = self.financial_risk_model.calculate_risk_score(self.risk_data)
        operational_risk_score = self.operational_risk_model.calculate_risk_score(self.risk_data)
        reputational_risk_score = self.reputational_risk_model.calculate_risk_score(self.risk_data)
        return financial_risk_score, operational_risk_score, reputational_risk_score

    def calculate_correlation_matrix(self):
        correlation_matrix = calculate_correlation_matrix(self.risk_data)
        return correlation_matrix

    def visualize_risk_heatmap(self):
        risk_heatmap = RiskHeatmap(self.correlation_matrix)
        risk_heatmap.plot()

    def visualize_risk_waterfall(self):
        risk_waterfall = RiskWaterfall(self.risk_scores)
        risk_waterfall.plot()

def main():
    risk_data = load_risk_data('data/risk_data.csv')
    risk_factors = load_risk_factors('data/risk_factors.json')
    multi_dim_risk_analysis = MultiDimRiskAnalysis(risk_data, risk_factors)
    risk_scores = multi_dim_risk_analysis.calculate_risk_scores()
    correlation_matrix = multi_dim_risk_analysis.calculate_correlation_matrix()
    multi_dim_risk_analysis.visualize_risk_heatmap()
    multi_dim_risk_analysis.visualize_risk_waterfall()

if __name__ == '__main__':
    main()
