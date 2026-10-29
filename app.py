import streamlit as st
import pandas as pd
import pickle 


# Load the trained model
with open("prophet_model.pkl", "rb") as file:
    model=pickle.load(file)

st.title("Melbourne Temperature Forecast")
st.write("Forecast daily minimum temperatures using our prohet model")

# Let the user choose the forecast period
st.subheader("Choose Forecast Period")

forecast_days = st.selectbox(
    "How many days do you want to forecast?",
    [7, 30, 90]
)

st.write(f"You selected a {forecast_days}-day forecast.")



# Generate forecast
future = pd.DataFrame({
    "ds": pd.date_range(
        "1991-01-01",
        periods=forecast_days,
        freq="D"
    )
})

forecast = model.predict(future)





st.subheader("🌡️ Temperature Forecast")

chart_data = forecast[["ds", "yhat"]].copy()
chart_data["ds"] = chart_data["ds"].dt.strftime("%d %b %Y")
chart_data = chart_data.set_index("ds")
chart_data.columns = ["Predicted Temperature (°C)"]

st.line_chart(chart_data)


