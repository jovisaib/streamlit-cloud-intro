# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Data Explorer!", page_icon="📊")

# Header
st.title("Data Explorer!!!")
st.write("A simple app to explore your data")

# Data generation
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.randn(100).cumsum()
    df = pd.DataFrame({"Date": dates, "Value": values})
    return df

# Sidebar
st.sidebar.header("Controls")
data_size = st.sidebar.slider("Sample Size", 10, 100, 50)

# Generate data
df = generate_data().iloc[:data_size]

# Display data
st.subheader("Raw Data")
st.dataframe(df)

# Visualization
st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Date"], df["Value"])
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True)
st.pyplot(fig)

# Download options
st.subheader("Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)