import joblib
import pandas as pd
model = joblib.load("car_model.pkl")
sample = pd.DataFrame({"Brand":[10],"model":[50],"Year":[2019],
                       "Age":[7],"kmDriven":[45000],
                       "Transmission":[1],"Owner":[0],"FuelType":[1]})
price = model.predict(sample)
print("Predicted price =" , int(price[0]))
