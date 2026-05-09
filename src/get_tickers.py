import pandas as pd
import requests
from io import StringIO

# Wikipedia maintains an up-to-date list of S&P 500 companies
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wikipedia blocks requests without a User-Agent header, so we set one
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

# Hand the HTML to pandas
tables = pd.read_html(StringIO(response.text))
sp500 = tables[0]

print("Columns available:", sp500.columns.tolist())
print(f"\nTotal companies: {len(sp500)}")
print(sp500[['Symbol', 'Security', 'GICS Sector']].head(10))

# Save to CSV
sp500.to_csv("data/sp500_tickers.csv", index=False)
print("\nSaved to data/sp500_tickers.csv")