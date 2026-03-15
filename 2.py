import pandas as pd

# Example portfolio data
data = {
    "Sector": ["Technology", "Healthcare", "Finance"],
    "Value": [50000, 30000, 20000]
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv("portfolio.csv", index=False)