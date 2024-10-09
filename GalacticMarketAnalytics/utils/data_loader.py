import pandas as pd

def load_data(config):
    db = config["database"]
    conn = psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["username"],
        password=db["password"]
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM galactic_market_data")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["date", "symbol", "open", "high", "low", "close", " volume"])
    return df
