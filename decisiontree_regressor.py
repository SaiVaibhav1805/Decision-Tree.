import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Tree Regressor", layout="wide")
st.title("📈 Decision Tree Regressor — Purchase Dataset")

st.info("""
ℹ️ **Note:** The Purchase dataset has categorical columns.
For regression, the target is **label-encoded** (No → 0, Yes → 1)
and the model predicts a continuous score between 0 and 1.
""")

# ── Load Data ──────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using built-in Purchase dataset.")
    data = {
        "Holiday":       ["No","No","No","yes","yes","yes","yes","No","yes","yes","yes","yes","yes","yes","yes","No","No","No","No","No","yes","yes","yes","yes","yes","yes","No","yes","No","No"],
        "Discount":      ["Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes"],
        "Free Delivery": ["Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No"],
        "Purchase":      ["Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes"]
    }
    df = pd.DataFrame(data)

st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Sidebar Config ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Configuration")
target_col = st.sidebar.selectbox("Target Column", df.columns.tolist(), index=len(df.columns)-1)
feature_cols = st.sidebar.multiselect("Feature Columns", [c for c in df.columns if c != target_col],
                                       default=[c for c in df.columns if c != target_col])

test_size         = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
max_depth         = st.sidebar.slider("Max Depth", 1, 10, 3)
criterion         = st.sidebar.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf  = st.sidebar.slider("Min Samples Leaf",  1, 10, 1)
random_state      = st.sidebar.number_input("Random State", 0, 100, 42)

if st.sidebar.button("🚀 Train Regressor"):
    if not feature_cols:
        st.error("Please select at least one feature column.")
    else:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Encode features
        encoders = {}
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].str.strip().str.lower())
                encoders[col] = le

        # Encode target as numeric (0/1)
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.str.strip().str.lower()).astype(float)
        class_names = le_target.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=int(random_state)
        )

        reg = DecisionTreeRegressor(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=int(random_state)
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        # ── Metrics ───────────────────────────────────────────────────────
        st.subheader("📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("RMSE",     f"{rmse:.4f}")
        col3.metric("MAE",      f"{mae:.4f}")
        col4.metric("MSE",      f"{mse:.4f}")

        # ── Actual vs Predicted ───────────────────────────────────────────
        st.subheader("🔁 Actual vs Predicted")
        result_df = pd.DataFrame({
            "Actual":    y_test,
            "Predicted": y_pred
        }).reset_index(drop=True)
        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_test, y_pred, color="steelblue", edgecolors="white", s=80, alpha=0.8)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # ── Residuals ─────────────────────────────────────────────────────
        st.subheader("📉 Residual Plot")
        residuals = y_test - y_pred
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(y_pred, residuals, color="coral", edgecolors="white", s=80, alpha=0.8)
        axes[0].axhline(0, color="black", lw=1.5, linestyle="--")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        sns.histplot(residuals, kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Residual Distribution")
        st.pyplot(fig2)

        # ── Feature Importance ────────────────────────────────────────────
        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": reg.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="magma", ax=ax3)
        ax3.set_title("Feature Importances")
        st.pyplot(fig3)

        # ── Tree Visualization ────────────────────────────────────────────
        st.subheader("🌲 Decision Tree Visualization")
        fig4, ax4 = plt.subplots(figsize=(16, 6))
        plot_tree(reg, feature_names=feature_cols,
                  filled=True, rounded=True, ax=ax4)
        st.pyplot(fig4)

        st.subheader("📝 Tree Rules (Text)")
        tree_rules = export_text(reg, feature_names=feature_cols)
        st.code(tree_rules)

        # ── Predict on New Input ──────────────────────────────────────────
        st.subheader("🔮 Predict on New Input")
        input_data = {}
        cols_input = st.columns(len(feature_cols))
        for i, col in enumerate(feature_cols):
            unique_vals = df[col].str.strip().str.lower().unique().tolist()
            input_data[col] = cols_input[i].selectbox(f"{col}", unique_vals)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if col in encoders:
                    input_df[col] = encoders[col].transform(input_df[col])
            pred_score = reg.predict(input_df)[0]
            pred_label = class_names[round(pred_score)] if 0 <= round(pred_score) < len(class_names) else "Unknown"
            st.success(f"✅ Predicted Score: **{pred_score:.4f}**")
            st.info(f"🏷️ Interpreted Label: **{pred_label.upper()}**  (0 = {class_names[0]}, 1 = {class_names[1]})")