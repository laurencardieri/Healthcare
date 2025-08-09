import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Online Healthcare Charge Predictor", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold;
    }
    .main-header {
        text-align: center;
        color: black;  
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">🩺 Online Healthcare Charge Predictor🩺</h1>', unsafe_allow_html=True)
st.markdown("#### Healthcare Predictor based on personal metrics")

@st.cache_resource
def load_model():
    model = joblib.load('model1.pkl')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.pkl')
    return scaler

model = load_model()
scaler = load_scaler()

@st.cache_data
def load_dataset():
    return pd.read_csv("insurance.csv")

df = load_dataset()

region_mean_encoded = {
    "Northeast": 13406.384516,
    "Northwest": 12417.575374,
    "Southeast": 14735.411438,
    "Southwest": 12346.937377
}

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Heathcare Predictor", "📊 Model Performance", "⚕️ Dataset", "🗃️ Data Exploration", "📈 Data Visualiation"])

with tab1:
    st.subheader("📋 Enter Patient Info")
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 65, 23)
            bmi = st.number_input("BMI", 10.0, 55.0, 25.0, step=0.1)
            children = st.selectbox("Number of Children", range(6))

        with col2:
            sex = st.selectbox("Sex", options=["Male", "Female"])
            smoker = st.selectbox("Smoker", options=["Yes", "No"])
            region = st.selectbox("Region", options=["Southwest", "Southeast", "Northwest", "Northeast"])

        submit = st.form_submit_button("Predict Charge")

    if submit:
        smoker_binary = 1 if smoker.lower() == "yes" else 0
        Gender_male = 1 if sex.lower() == "male" else 0
        region_encoded = region_mean_encoded[region]

        input_df = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "smoker": smoker_binary,
            "Gender_male": Gender_male,
            "region_encoded": region_encoded
        }])

        input_scaled = scaler.transform(input_df)

        try:
            pred = model.predict(input_scaled)[0]
            preds = [model.predict(input_scaled)[0] for _ in range(100)]
            ci_low = np.percentile(preds, 2.5)
            ci_high = np.percentile(preds, 97.5)
            st.success(f"💵 Predicted Charge: **${pred:,.2f}**")
            st.info(f"95% Confidence Interval: ${ci_low:,.2f} - ${ci_high:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("📊 Model Performance")

    df_encoded = df.copy()
    df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
    df_encoded['Gender_male'] = df_encoded['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df_encoded['region_encoded'] = df_encoded['region'].str.title().map(region_mean_encoded)

    X = df_encoded.drop(columns=['charges', 'sex', 'region'])
    y = df_encoded['charges']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:,.2f}")
    col2.metric("RMSE", f"${rmse:,.2f}")
    col3.metric("R²", f"{r2:.2f}")

    fig2 = px.scatter(x=y, y=y_pred, labels={"x": "Actual Charges", "y": "Predicted Charges"}, title="Actual vs Predicted Charges")
    st.plotly_chart(fig2)

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig3 = px.bar(
            feature_importance,
            title="Feature Importance",
            labels={'value': 'Importance', 'index': 'Feature'},
            orientation='h'
        )
        fig3.update_layout(yaxis=dict(tickmode='array', tickvals=list(range(len(feature_importance))), ticktext=feature_importance.index))
        st.plotly_chart(fig3)

with tab3:
    st.subheader("⚕️Dataset")
    df_encoded = df.copy()
    df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
    df_encoded['Gender_male'] = df_encoded['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df_encoded['region_encoded'] = df_encoded['region'].str.title().map(region_mean_encoded)
    st.dataframe(df_encoded.head(21))

    st.write("Credit: willian oliveira. (https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance)")

with tab4:
    st.subheader("Summary Statistics")
    st.write(df.describe())

with tab5:
    fig = px.box(df, x="region", y="charges", color="region", title="Charges by Region")
    st.plotly_chart(fig)

    st.subheader("Correlation with Charges")

    df_encoded = df.copy()
    df_encoded['smoker'] = df_encoded['smoker'].map({'yes': 1, 'no': 0})
    df_encoded['Gender_male'] = df_encoded['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df_encoded['region_encoded'] = df_encoded['region'].str.title().map(region_mean_encoded)
    df_encoded = df_encoded.drop(columns=['sex', 'region'])

    series = df_encoded.corr()['charges'].drop('charges')
    charges_cm = series.to_frame(name='charges')

    plt.figure(figsize=(4, 5))
    sns.heatmap(charges_cm, annot=True, cmap='coolwarm', center=0, square=False, linewidths=0.5)
    plt.title('Correlation of Factors with Charges')

    st.pyplot(plt.gcf())
