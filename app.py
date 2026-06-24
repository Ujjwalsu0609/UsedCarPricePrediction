import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from feature_importance import plot_feature_importance

# Load model
model = joblib.load("car_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load encoders
brand_encoder = joblib.load("brand_encoder.pkl")
model_encoder = joblib.load("model_encoder.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")
transmission_encoder = joblib.load("transmission_encoder.pkl")
owner_encoder = joblib.load("owner_encoder.pkl")
df = pd.read_csv("used_car_dataset.csv")
df["AskPrice"] = (
    df["AskPrice"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["AskPrice"] = pd.to_numeric(df["AskPrice"],errors="coerce")
brand_model_map = (df.groupby("Brand")["model"].unique().apply(list).to_dict())
current_year = datetime.now().year

st.title("🚗 Used car price prediction")


col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        sorted(df["Brand"].unique())
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=2000,
        max_value=2026,
        value=2020
    )

    fuel = st.selectbox(
        "Fuel Type",
        fuel_encoder.classes_
    )

with col2:
    available_models = sorted(
        df[df["Brand"] == brand]["model"].unique()
    )

    car_model = st.selectbox(
        "Model",
        available_models
    )

    km = st.number_input(
        "KM Driven",
        min_value=0,
        value=50000
    )

    transmission = st.selectbox(
        "Transmission",
        transmission_encoder.classes_
    )

owner = st.selectbox(
    "Owner",
    owner_encoder.classes_
)

if st.button("Predict Price"):

    age = current_year - year

    sample = pd.DataFrame({
        "Brand": [brand_encoder.transform([brand])[0]],
        "model": [model_encoder.transform([car_model])[0]],
        "Year": [year],
        "Age": [age],
        "kmDriven": [km],
        "Transmission": [transmission_encoder.transform([transmission])[0]],
        "Owner": [owner_encoder.transform([owner])[0]],
        "FuelType": [fuel_encoder.transform([fuel])[0]]
    })

    # Validation
    if year > current_year:
        st.error("Invalid manufacturing year")
        st.stop()

    if km < 0:
        st.error("KM Driven cannot be negative")
        st.stop()

    # Prediction
    predicted_price = model.predict(sample)[0]

    # Dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Price", f"₹{int(predicted_price):,}")

    with col2:
        st.metric("Car Age", age)

    with col3:
        st.metric("KM Driven", f"{km:,}")

    # Analytics
    col4, col5 = st.columns(2)

    with col4:
        st.metric("Total Cars", len(df))

    with col5:
        st.metric(
            "Average Price",
            f"₹{df['AskPrice'].mean():,.0f}"
        )

    # Feature Importance
    fig = plot_feature_importance(model, feature_names)

    st.subheader("Feature Importance")
    st.pyplot(fig)
    brand_price = (df.groupby("Brand")["AskPrice"].mean().sort_values(ascending=False).head(10))
    st.subheader("Top 10 Brands by Average Price")
    st.bar_chart(brand_price)

    # Final Result
    st.success(
        f"Estimated Price: ₹{int(predicted_price):,}"
    )