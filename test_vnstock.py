from vnstock import Quote

quote = Quote(symbol="ACB", source="VCI")
df = quote.history(start="2024-01-01", end="2024-03-01", interval="1D")

print(df.head())
print(df.shape)