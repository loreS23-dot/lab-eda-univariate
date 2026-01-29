import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Para que display no rompa en terminal
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)

# ==== CARGA ====
df = pd.read_csv(r"C:\Users\marcl\Downloads\amz_uk_price_prediction_dataset.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(5))

# ==== LIMPIEZA MINIMA ====
# price a numérico
if df["price"].dtype == "object":
    df["price"] = (
        df["price"].astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# stars a numérico
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

print("\nMissing ratio (category/price/stars):")
print(df[["category","price","stars"]].isna().mean())

# =========================
# PART 1: CATEGORIES
# =========================
print("\n====================")
print("PART 1: CATEGORIES")
print("====================")

cat_freq = df["category"].value_counts(dropna=False)
print("\nFrequency table (top 15):")
print(cat_freq.head(15))

top5 = df["category"].value_counts().head(5)
print("\nTOP 5 categorías:")
print(top5)

top_n = 15
top_cats = df["category"].value_counts().head(top_n)

plt.figure(figsize=(10, 6))
top_cats.sort_values().plot(kind="barh")
plt.title(f"Top {top_n} categorías por número de listings")
plt.xlabel("Número de productos")
plt.ylabel("Categoría")
plt.tight_layout()
plt.show()

pie_n = 6
pie_data = df["category"].value_counts().head(pie_n)

plt.figure(figsize=(7, 7))
plt.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%")
plt.title(f"Proporción de listings (Top {pie_n} categorías)")
plt.show()

# =========================
# PART 2: PRICING
# =========================
print("\n====================")
print("PART 2: PRICING")
print("====================")

price = df["price"].dropna()

mean_price = price.mean()
median_price = price.median()
mode_price = price.mode().iloc[0] if not price.mode().empty else np.nan

print("\nCentralidad (price):")
print("Mean:", mean_price)
print("Median:", median_price)
print("Mode:", mode_price)

var_price = price.var()
std_price = price.std()
range_price = price.max() - price.min()
iqr_price = price.quantile(0.75) - price.quantile(0.25)

print("\nDispersión (price):")
print("Varianza:", var_price)
print("Std:", std_price)
print("Rango:", range_price)
print("IQR:", iqr_price)

print("\nPercentiles (price):")
print(price.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

plt.figure(figsize=(10, 5))
plt.hist(price, bins=50)
plt.title("Distribución de precios (sin recorte)")
plt.xlabel("Precio")
plt.ylabel("Número de productos")
plt.tight_layout()
plt.show()

p99 = price.quantile(0.99)
price_clip = price[price <= p99]
print("\nP99 precio:", p99, "| recortados:", len(price)-len(price_clip))

plt.figure(figsize=(10, 5))
plt.hist(price_clip, bins=50)
plt.title("Distribución de precios (hasta percentil 99)")
plt.xlabel("Precio")
plt.ylabel("Número de productos")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.boxplot(price, vert=False)
plt.title("Boxplot de precios (outliers)")
plt.xlabel("Precio")
plt.tight_layout()
plt.show()

# =========================
# PART 3: RATINGS (stars)
# =========================
print("\n====================")
print("PART 3: RATINGS (stars)")
print("====================")

stars = df["stars"].dropna()

mean_stars = stars.mean()
median_stars = stars.median()
mode_stars = stars.mode().iloc[0] if not stars.mode().empty else np.nan

print("\nCentralidad (stars):")
print("Mean:", mean_stars)
print("Median:", median_stars)
print("Mode:", mode_stars)

var_stars = stars.var()
std_stars = stars.std()
iqr_stars = stars.quantile(0.75) - stars.quantile(0.25)

print("\nDispersión (stars):")
print("Varianza:", var_stars)
print("Std:", std_stars)
print("IQR:", iqr_stars)

skew_stars = skew(stars)
kurt_stars = kurtosis(stars, fisher=True)

print("\nForma (stars):")
print("Skewness:", skew_stars)
print("Kurtosis (exceso):", kurt_stars)

plt.figure(figsize=(10, 5))
plt.hist(stars, bins=20)
plt.title("Distribución de ratings (stars)")
plt.xlabel("Stars")
plt.ylabel("Número de productos")
plt.tight_layout()
plt.show()
