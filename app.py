
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="Melbourne Temperature Forecast",
    page_icon="",
    layout="wide"
)

# Load trained model
with open("prophet_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load historical data
df = pd.read_csv(
    "daily-minimum-temperatures-in-me.csv",
    skipfooter=2,
    engine="python"
)

df.rename(columns={
    "Daily minimum temperatures in Melbourne, Australia, 1981-1990":
        "Temperature"
}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df["Temperature"] = pd.to_numeric(
    df["Temperature"], errors="coerce"
)
df["Temperature"] = df["Temperature"].interpolate()

# Dashboard heading
st.title(" Melbourne Temperature Forecast")
st.write(
    "Explore historical daily minimum temperatures "
    "and generate forecasts using Prophet."
)

st.divider()

# Historical overview
st.subheader(" Historical Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Historical Records", f"{len(df):,}")

with col2:
    st.metric(
        "Average Temperature",
        f"{df['Temperature'].mean():.1f} °C"
    )

with col3:
    st.metric(
        "Date Range",
        f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}"
    )

# Historical chart
st.subheader("Historical Temperature")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    df["Date"], df["Temperature"],
    label="Actual Temperature", linewidth=0.8
)
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# Forecast settings
st.subheader("Generate Future Forecast")

forecast_days = st.selectbox(
    "Choose forecast duration",
    [7, 30, 90, 365]
)

# Dataset ends on 31 December 1990
last_date = df["Date"].max()

future = pd.DataFrame({
    "ds": pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )
})

forecast = model.predict(future)

# Forecast summary
col1, col2 = st.columns(2)

with col1:
    st.metric("Forecast Days", forecast_days)

with col2:
    st.metric(
        "Average Predicted Temperature",
        f"{forecast['yhat'].mean():.1f} °C"
    )

# Forecast chart with uncertainty interval
st.subheader("Temperature Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast["ds"], forecast["yhat"],
    label="Prophet Forecast",
    linewidth=2
)

ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2,
    label="Uncertainty Interval"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Forecast table
st.subheader("Forecast Results")

forecast_table = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].rename(columns={
    "ds": "Date",
    "yhat": "Predicted Temperature (°C)",
    "yhat_lower": "Lower Estimate (°C)",
    "yhat_upper": "Upper Estimate (°C)"
})

st.dataframe(forecast_table, use_container_width=True)

# Download forecast
csv = forecast_table.to_csv(index=False).encode("utf-8")

st.download_button(
    label=" Download Forecast CSV",
    data=csv,
    file_name="melbourne_temperature_forecast.csv",
    mime="text/csv"
)

st.caption(
    "Forecasts are model estimates, not guaranteed future temperatures."
)
