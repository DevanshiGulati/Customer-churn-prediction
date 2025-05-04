import pandas as pd
import numpy as np

np.random.seed(42)
num_records = 100000

data = {
    "CustomerID": [f"CUST{str(i).zfill(5)}" for i in range(1, num_records + 1)],
    "AccountAge": np.random.randint(1, 60, num_records),  # months
    "UsageFrequency": np.random.randint(1, 30, num_records),  # sessions/month
    "MonthlySpend": np.round(np.random.uniform(10.0, 500.0, num_records), 2),
    "SupportCalls": np.random.poisson(2, num_records),
    "Churn": np.random.choice(["Yes", "No"], size=num_records, p=[0.2, 0.8])
}

df = pd.DataFrame(data)
df.to_csv("DATA/raw_customer_data.csv", index=False)

print("✅ Synthetic customer churn dataset generated.")
