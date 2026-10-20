import pandas as pd
import json

# Load processed data for demo
customer = pd.read_csv('data/processed/customer_data.csv')
product = pd.read_csv('data/processed/product_data.csv')
cancel = pd.read_csv('data/processed/cancellations.csv')
clean = pd.read_parquet('data/processed/cleaned_transactions.parquet')

metrics = {}

# Executive metrics
metrics['total_revenue'] = round(float(clean['TotalAmount'].sum()), 2)
metrics['total_orders'] = int(clean['InvoiceNo'].nunique())
metrics['total_customers'] = int(customer.shape[0])
metrics['total_products'] = int(product.shape[0])
metrics['total_units'] = int(clean['Quantity'].sum())
metrics['total_countries'] = int(clean['Country'].nunique())
metrics['total_cancellations'] = int(cancel.shape[0])
metrics['cancel_pct'] = round(cancel.shape[0] / (clean.shape[0] + cancel.shape[0]) * 100, 2)

# Top 10 products by revenue
top_prods = product.head(10)[['StockCode','Description','TotalRevenue','TotalQuantitySold','UniqueCustomers','TotalInvoices']].copy()
top_prods['Description'] = top_prods['Description'].str[:50]
metrics['top_products'] = top_prods.to_dict('records')

# Top 10 customers by spend
top_custs = customer.nlargest(10, 'TotalSpend')[['CustomerID','TotalSpend','TotalOrders','TotalUnits','UniqueProducts']].copy()
metrics['top_customers'] = top_custs.to_dict('records')

# Country revenue distribution (top 10)
country_rev = clean.groupby('Country').agg(
    revenue=('TotalAmount', 'sum'),
    orders=('InvoiceNo', 'nunique'),
    customers=('CustomerID', 'nunique')
).sort_values('revenue', ascending=False).head(10).reset_index()
metrics['top_countries'] = country_rev.to_dict('records')

# Monthly revenue trend
clean['YearMonth'] = pd.to_datetime(clean['InvoiceDate']).dt.to_period('M').astype(str)
monthly = clean.groupby('YearMonth').agg(
    revenue=('TotalAmount', 'sum'),
    orders=('InvoiceNo', 'nunique'),
    units=('Quantity', 'sum')
).reset_index()
metrics['monthly_trend'] = monthly.to_dict('records')

# Customer spending distribution buckets
bins = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '50K+']
customer['spend_bucket'] = pd.cut(customer['TotalSpend'], bins=bins, labels=labels)
spend_dist = customer['spend_bucket'].value_counts().sort_index().to_dict()
metrics['spend_distribution'] = {str(k): int(v) for k, v in spend_dist.items()}

# Product quantity distribution (top 20 by quantity)
top_qty = product.nlargest(20, 'TotalQuantitySold')[['StockCode','Description','TotalQuantitySold','TotalRevenue']].copy()
top_qty['Description'] = top_qty['Description'].str[:35]
metrics['top_products_qty'] = top_qty.to_dict('records')

# Cancellation by country (top 5)
cancel_country = cancel.groupby('Country').size().sort_values(ascending=False).head(5).reset_index()
cancel_country.columns = ['country', 'count']
metrics['cancel_by_country'] = cancel_country.to_dict('records')

# Customer order frequency distribution
freq_bins = [0, 1, 3, 5, 10, 20, float('inf')]
freq_labels = ['1', '2-3', '4-5', '6-10', '11-20', '20+']
customer['order_bucket'] = pd.cut(customer['TotalOrders'], bins=freq_bins, labels=freq_labels)
order_dist = customer['order_bucket'].value_counts().sort_index().to_dict()
metrics['order_distribution'] = {str(k): int(v) for k, v in order_dist.items()}

# Save as JSON for the demo website
with open('demo_data.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

print("Demo data extracted successfully!")
print(f"Revenue: {metrics['total_revenue']:,.2f}")
print(f"Orders: {metrics['total_orders']:,}")
print(f"Customers: {metrics['total_customers']:,}")
print(f"Products: {metrics['total_products']:,}")
print(f"Countries: {metrics['total_countries']}")
print(f"Monthly data points: {len(metrics['monthly_trend'])}")
