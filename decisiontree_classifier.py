import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Tree Classifier", layout="wide")
st.title("🌳 Decision Tree Classifier — Purchase Dataset")

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

test_size     = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
max_depth     = st.sidebar.slider("Max Depth", 1, 10, 3)
criterion     = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf  = st.sidebar.slider("Min Samples Leaf",  1, 10, 1)
random_state  = st.sidebar.number_input("Random State", 0, 100, 42)

if st.sidebar.button("🚀 Train Classifier"):
    if not feature_cols:
        st.error("Please select at least one feature column.")
    else:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Encode
        encoders = {}
        for col in X.columns:
            if X[col].dtype == object:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].str.strip().str.lower())
                encoders[col] = le

        le_target = LabelEncoder()
        y = le_target.fit_transform(y.str.strip().str.lower())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=int(random_state)
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # ── Metrics ───────────────────────────────────────────────────────
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("Train Size", len(X_train))
        col3.metric("Test Size",  len(X_test))

        st.text("Classification Report:")
        report = classification_report(y_test, y_pred,
                                       target_names=le_target.classes_, zero_division=0)
        st.code(report)

        # ── Confusion Matrix ──────────────────────────────────────────────
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # ── Feature Importance ────────────────────────────────────────────
        st.subheader("📌 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature":   feature_cols,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax2)
        ax2.set_title("Feature Importances")
        st.pyplot(fig2)

        # ── Tree Visualization ────────────────────────────────────────────
        st.subheader("🌲 Decision Tree Visualization")
        fig3, ax3 = plt.subplots(figsize=(16, 6))
        plot_tree(clf, feature_names=feature_cols,
                  class_names=le_target.classes_,
                  filled=True, rounded=True, ax=ax3)
        st.pyplot(fig3)

        st.subheader("📝 Tree Rules (Text)")
        tree_rules = export_text(clf, feature_names=feature_cols)
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
            pred = clf.predict(input_df)
            pred_label = le_target.inverse_transform(pred)[0]
            prob = clf.predict_proba(input_df)[0]
            st.success(f"✅ Prediction: **{pred_label.upper()}**")
            prob_df = pd.DataFrame({"Class": le_target.classes_, "Probability": prob})
            st.dataframe(prob_df)