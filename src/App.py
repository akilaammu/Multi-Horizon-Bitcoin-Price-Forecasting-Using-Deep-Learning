import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

from tensorflow.keras.models import load_model
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Bitcoin Forecasting System",
    layout="wide"
)


# Title
st.title("₿ Bitcoin Price Forecasting System")


# Load dataset
@st.cache_data
def load_data():

    df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Bitcoin Historical Data.csv"
    )

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by Date
    df = df.sort_values("Date")

    # Clean Price column
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Remove missing values
    df = df.dropna()

    return df


# Load model
@st.cache_resource
def load_prediction_model():

    model = load_model(
        "models/Best_Bitcoin_Model.keras"
    )

    return model


# Load scaler
@st.cache_resource
def load_scaler():

    with open(
        "scaler/bitcoin_scaler.pkl",
        "rb"
    ) as file:

        scaler = pickle.load(file)

    return scaler



df = load_data()
model = load_prediction_model()
scaler = load_scaler()


#st.success("Model and data loaded successfully")

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        "Dashboard",
        "Prediction",
        "Model Performance",
        "Actual vs Predicted",
        "Error Distribution",
        "Forecast Horizon",
        "Training Loss"
    ]
)

# Prediction Function
def predict_future(model, scaler, recent_prices):
    
    # Convert to NumPy array
    recent_prices = np.array(recent_prices, dtype=np.float32)

    # Scale
    scaled_prices = scaler.transform(
        recent_prices.reshape(-1, 1)
    )

    # Shape: (1, 60, 1)
    X_input = scaled_prices.reshape(1, 60, 1)

    # Predict
    prediction = model.predict(X_input, verbose=0)

    # prediction shape -> (1, 7)
    prediction = prediction.reshape(-1, 1)

    # Convert back to original prices
    prediction = scaler.inverse_transform(prediction)

    return prediction.flatten()

# ---------------------- Dashboard ----------------------
if page == "Dashboard":
    st.header("📊 Dashboard")
    # KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latest Bitcoin Price",
            f"${df['Price'].iloc[-1]:,.2f}"
        )

    with col2:
        st.metric(
            "Best Model",
            "RNN"
        )

    with col3:
        st.metric(
            "Forecast Horizon",
            "7 Days"
        )

    st.markdown("---")
    st.subheader("📈 Historical Bitcoin Price")

    fig = px.line(
        df,
        x="Date",
        y="Price",
        title="Historical Bitcoin Closing Price"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.tail(10))

    st.subheader("📊 Dataset Statistics")
    st.write(df["Price"].describe())


# ---------------------- Prediction ----------------------
elif page == "Prediction":

    st.header("🔮 Bitcoin Price Forecast")

    st.subheader("Last 60 Days Closing Prices")
    st.line_chart(df["Price"].tail(60))

    if st.button("Predict Future Prices"):

        latest_prices = df["Price"].tail(60).values

        predictions = predict_future(
            model,
            scaler,
            latest_prices
        )

        last_date = df["Date"].iloc[-1]

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(predictions)
        )

        prediction_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": predictions
        })

        #st.success("Prediction completed successfully!")

        st.subheader("Predicted Prices")
        st.dataframe(prediction_df)

        fig = px.line(
            prediction_df,
            x="Date",
            y="Predicted Price",
            markers=True,
            title="Future Bitcoin Price Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)
elif page == "Model Performance":
    
    st.header("📊 Model Performance Comparison")

    comparison_df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Bitcoin_Model_Comparison.csv"
    )

    st.dataframe(comparison_df)

    fig = px.bar(
        comparison_df,
        x="Model",
        y="MAE",
        color="Model",
        title="Model Comparison based on MAE"
    )

    st.plotly_chart(fig, use_container_width=True)
    # Best Model
    best_model = comparison_df.loc[
        comparison_df["MAE"].idxmin()
    ]

    st.success(
        f"🏆 Best Model: {best_model['Model']}"
    ) 
elif page == "Actual vs Predicted":
    
    st.header("📈 Actual vs Predicted Price")

    actual_df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Actual_vs_Predicted.csv"
    )

    st.dataframe(actual_df.head())

    fig = px.line(
        actual_df,
        y=["Actual Price", "Predicted Price"],
        title="Actual vs Predicted Bitcoin Price"
    )

    st.plotly_chart(fig, use_container_width=True)
elif page == "Error Distribution":
    
    st.header("📊 Prediction Error Distribution")

    error_df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Prediction_Error.csv"
    )

    st.dataframe(error_df.head())

    fig = px.histogram(
        error_df,
        x="Error",
        nbins=40,
        title="Prediction Error Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
elif page == "Forecast Horizon":
    
    st.header("📉 Forecast Horizon Comparison")

    horizon_df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Forecast_Horizon.csv"
    )

    st.dataframe(horizon_df)

    fig = px.bar(
        horizon_df,
        x="Horizon",
        y="Best MAE",
        color="Horizon",
        title="Forecast Horizon Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)
elif page == "Training Loss":
    
    st.header("📉 Training vs Validation Loss")

    loss_df = pd.read_csv(
        r"C:/Users/Thang/Desktop/MiniProject4/DataSet/Training_Loss.csv"
    )

    st.dataframe(loss_df.head())

    fig = px.line(
        loss_df,
        x="Epoch",
        y=["Training Loss", "Validation Loss"],
        title="Training History"
    )

    st.plotly_chart(fig, use_container_width=True)                