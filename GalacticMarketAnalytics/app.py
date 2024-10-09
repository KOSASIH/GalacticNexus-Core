import config
from models import MarketTrendsModel, SentimentAnalysisModel, PredictiveModel
from utils import data_loader, data_preprocessor, feature_engineer

def main():
    config = config.config
    df = data_loader.load_data(config)
    df = data_preprocessor.preprocess_data(df)
    df = feature_engineer.engineer_features(df)

    market_trends_model = MarketTrendsModel(config)
    sentiment_analysis_model = SentimentAnalysisModel(config)
    predictive_model = PredictiveModel(config)

    market_trends_model.train(df, df["close"])
    sentiment_analysis_model.train(df, df["text"])
    predictive_model.train(df, df["close"])

    market_trends_pred = market_trends_model.predict(df)
    sentiment_analysis_pred = sentiment_analysis_model.predict(df)
    predictive_pred = predictive_model.predict(df)

    print("Market Trends Prediction:", market_trends_pred)
    print("Sentiment Analysis Prediction:", sentiment_analysis_pred)
    print("Predictive Model Prediction:", predictive_pred)

if __name__ == "__main__":
    main()
