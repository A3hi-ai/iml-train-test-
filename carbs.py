

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


rng = np.random.default_rng(42)   
N = 500

age = rng.integers(18, 66, N)
gender = rng.choice(["male", "female"], N)
is_male = (gender == "male")


height = np.where(is_male, rng.normal(176, 7, N), rng.normal(163, 6.5, N))
bmi = np.where(is_male, rng.normal(25.5, 3.5, N), rng.normal(24.5, 4.0, N))
weight = bmi * (height / 100) ** 2

duration = rng.integers(5, 61, N)                  


heart_rate = 75 + 0.9 * duration - 0.2 * (age - 18) + rng.normal(0, 5, N)
heart_rate = np.clip(heart_rate, 70, 190)


body_temp = 37.0 + 0.045 * duration + rng.normal(0, 0.2, N)
body_temp = np.clip(body_temp, 36.8, 41.0)


kcal_per_min = np.where(
    is_male,
    (-55.0969 + 0.6309 * heart_rate + 0.1988 * weight + 0.2017 * age) / 4.184,
    (-20.4022 + 0.4472 * heart_rate - 0.1263 * weight + 0.0740 * age) / 4.184,
)
energy_kcal = np.clip(kcal_per_min, 1, None) * duration


intensity = heart_rate / (220 - age)
carb_fraction = np.clip(0.30 + (intensity - 0.4) * 1.0, 0.25, 0.85)


carbs = energy_kcal * carb_fraction / 4.0 + rng.normal(0, 2.0, N)
carbs = np.clip(carbs, 0.5, None)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Height": height.round(1),
    "Weight": weight.round(1),
    "Duration": duration,
    "Heart_Rate": heart_rate.round(0).astype(int),
    "Body_Temp": body_temp.round(1),
    "Carbs": carbs.round(1),          
})

df.to_csv("carbs_dataset.csv", index=False)
print("Dataset created and saved to carbs_dataset.csv")


print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Shape ---", df.shape)
print("\n--- Info ---")
df.info()
print("\n--- Summary statistics ---")
print(df.describe().round(2))
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Gender counts ---")
print(df["Gender"].value_counts())


df_num = df.copy()
df_num["Gender"] = df_num["Gender"].map({"male": 1, "female": 0})

plt.figure(figsize=(9, 7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()


df_num.hist(bins=25, figsize=(12, 8), edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.show()


X = df_num.drop(columns="Carbs")
y = df_num["Carbs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model coefficients ---")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:12s} {coef:8.3f}")
print(f"{'Intercept':12s} {model.intercept_:8.3f}")


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Evaluation on test set ---")
print(f"R-squared : {r2:.4f}")
print(f"MAE       : {mae:.2f} g of carbs")


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Carbs (g)")
plt.ylabel("Predicted Carbs (g)")
plt.title(f"Actual vs Predicted (R2 = {r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

new_person = pd.DataFrame([{
    "Age": 25, "Gender": 1, "Height": 178, "Weight": 75,
    "Duration": 30, "Heart_Rate": 110, "Body_Temp": 38.5,
}])
print(f"\nPredicted carbs burned for example person: "
      f"{model.predict(new_person)[0]:.1f} g")
