import unittest
from ai_trading_engine import AITradingEngine

class TestAITradingEngine(unittest.TestCase):
    def test_train(self):
        engine = AITradingEngine()
        engine.train('data/training_data.csv')
        self.assertTrue(engine.model is not None)

    def test_predict(self):
        engine = AITradingEngine()
        engine.train('data/training_data.csv')
        prediction = engine.predict('data/testing_data.csv')
        self.assertTrue(prediction is not None)
