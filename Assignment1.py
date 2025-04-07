import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📊 Asset Allocation Optimizer", layout="wide")
st.title("📊 Asset Allocation Insights for Maximizing 3-Year Return")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("tradexa_dataset.csv")
    df = df.dropna(subset=["3YrReturn%", "Market Cap", "Type", "Risk"])
    df["3YrReturn%"] = df["3YrReturn%"].astype(str).str.replace("%", "").astype(float)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
selected_risk = st.sidebar.multiselect("Select Risk Level:", df["Risk"].unique(), default=df["Risk"].unique())
selected_type = st.sidebar.multiselect("Select Type:", df["Type"].unique(), default=df["Type"].unique())
selected_marketcap = st.sidebar.multiselect("Select Market Cap:", df["Market Cap"].unique(), default=df["Market Cap"].unique())

filtered_df = df[
    df["Risk"].isin(selected_risk) &
    df["Type"].isin(selected_type) &
    df["Market Cap"].isin(selected_marketcap)
]

st.subheader("Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True))

# Aggregated Visuals
st.subheader("📈 Average Return by Category")
col1, col2 = st.columns(2)

with col1:
    st.write("**By Risk**")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Risk", y="3YrReturn%", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("**By Market Cap**")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Market Cap", y="3YrReturn%", ax=ax)
    st.pyplot(fig)

# Model Prediction
st.subheader("🧠 Predicting 3YrReturn%")
df_model = df[["Market Cap", "Type", "Risk", "3YrReturn%"]].copy()
df_encoded = pd.get_dummies(df_model.drop("3YrReturn%", axis=1))
y = df_model["3YrReturn%"]
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.markdown(f"**Model R² Score:** {r2:.2f}")
st.markdown(f"**Mean Squared Error:** {mse:.2f}")

# Feature Importances
feature_importance_df = pd.DataFrame({
    "Feature": df_encoded.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.subheader("📊 Feature Importances")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=feature_importance_df, y="Feature", x="Importance", ax=ax)
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | AI Data Scientist Project")
