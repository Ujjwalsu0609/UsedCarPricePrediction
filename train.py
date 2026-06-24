import pandas as pd
import joblib
df = pd.read_csv("cleaned_used_car_dataset.csv")
df["AskPrice"] = (
    df["AskPrice"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["AskPrice"] = pd.to_numeric(df["AskPrice"])
df["AskPrice"] = pd.to_numeric(df["AskPrice"],errors="coerce")
df = df.dropna()
print(df.isnull().sum())
x = df.drop("AskPrice", axis=1)
y = df["AskPrice"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor (n_estimators=100,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)
print("R2 score =", score)
feature_names = x.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(model,"car_model.pkl")
print("model saved successfully")