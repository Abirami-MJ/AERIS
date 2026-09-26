import pandas as pd

# Try reading with semicolon separator
df = pd.read_csv(
    "final_reports_2016-23_cons_2024-12-24.csv",
    sep=';',
    encoding='utf-8',
    low_memory=False
)
# basic data info 
df.info()

#initial data inspection
print(df.head())
print(df.shape)

print("="*60)
print("Dataset Shape")
print(df.shape)

print("="*60)
print("Columns")
print(df.columns.tolist())

print("="*60)
print("Data Types")
print(df.dtypes)

print("="*60)
print(df.head())

print("="*60)
print(df.tail())

# Missing values 
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# percentage calc
missing_percent = (df.isnull().sum()/len(df))*100
missing_percent = missing_percent[missing_percent>0].sort_values(ascending=False)
print(missing_percent)

# Duplicate values 
print("Duplicate Rows :", df.duplicated().sum())

# Summary 
df.describe(include='all').T

#EDA Reports 
eda = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing": df.isnull().sum(),
    "Missing %": (df.isnull().sum()/len(df))*100,
    "Unique": df.nunique()
})

eda.to_csv("EDA_Report.csv", index=False)