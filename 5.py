import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('products.csv')

# Clean data
df.columns = df.columns.str.strip().str.lower()
df = df.dropna().drop_duplicates()

print("Columns in dataset:")
print(df.columns)

# -------- ML PART --------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

target_col = numeric_cols[-1]

X = df[numeric_cols].drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredictions (first 5):")
print(y_pred[:5])

# -------- Helper Function --------
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# -------- RQ1: Avg Price by Category --------
rq1 = df.groupby('product category')['price'].mean().reset_index()
rq1['price'] = normalize(rq1['price'])
rq1 = rq1.sort_values(by='price', ascending=False)

rq1.to_csv('RQ1_table.csv', index=False)

plt.figure()
sns.barplot(data=rq1, x='product category', y='price', width=0.4)
plt.title('Normalized Avg Price by Category')
plt.xticks(rotation=30)
plt.savefig('RQ1_figure.pdf')
plt.show()


# -------- RQ5: Top 5 Expensive Products --------
rq5 = df.sort_values(by='price', ascending=False).head(5)

rq5.to_csv('RQ5_table.csv', index=False)

plt.figure()
sns.barplot(data=rq5, x='product name', y='price', width=0.4)
plt.title('Top 5 Expensive Products')
plt.xticks(rotation=45)
plt.savefig('RQ5_figure.pdf')
plt.show()