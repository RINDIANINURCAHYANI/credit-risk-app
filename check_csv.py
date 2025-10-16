import pandas as pd

df = pd.read_csv("data/default_of_credit_card_clients.csv")
df.columns = df.columns.str.strip()
df = df.rename(columns={"default.payment.next.month": "default"})
df['default'] = df['default'].astype(str).str.strip()
df['default'] = df['default'].astype(int)

print(df.head())
