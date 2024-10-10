class Market:
    def __init__(self, market_index, interest_rate, inflation_rate):
        self.market_index = market_index
        self.interest_rate = interest_rate
        self.inflation_rate = inflation_rate

    def get_market_data(self):
        # Retrieve market data from a data source (e.g. API, CSV file)
        # For demonstration purposes, we'll use a sample CSV file
        market_data = pd.read_csv("market_data.csv")
        return market_data
