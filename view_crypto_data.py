import pandas as pd

print("🚀 Cryptocurrency Data Sample Viewer")
print("=" * 50)

cryptos = ['BTC', 'ETH', 'SOL']

for crypto in cryptos:
    filename = f'crypto_data/{crypto}_6year_data.csv'
    
    try:
        df = pd.read_csv(filename)
        
        print(f"\n💰 {crypto} Data Sample:")
        print("-" * 30)
        print(df.head())
        print(f"\n📊 Summary for {crypto}:")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   • Latest price: ${df['close_usd'].iloc[-1]:,.2f}")
        print(f"   • Highest price: ${df['high_usd'].max():,.2f}")
        print(f"   • Lowest price: ${df['low_usd'].min():,.2f}")
        
    except Exception as e:
        print(f"❌ Error reading {crypto} data: {e}")

print("\n🎉 All crypto data files are ready!")
print("\nColumns in each file:")
print("• date - Trading date")
print("• open_usd - Opening price in USD")
print("• high_usd - Highest price of the day in USD")
print("• low_usd - Lowest price of the day in USD") 
print("• close_usd - Closing price in USD")
print("• volume - Trading volume")