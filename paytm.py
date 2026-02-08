import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Paytm Churn & Financial Dashboard", layout="wide")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    # basic cleanup
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize some text columns
    for col in ["housing", "payment_type", "zodiac_sign"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # fix weird missing placeholders
    if "housing" in df.columns:
        df["housing"] = df["housing"].replace({"na": np.nan, "nan": np.nan})

    return df


DATA_PATH = r"C:\Users\Haresh L\OneDrive\Documents\decipher 5.0\698627992e03e_Round_2_dataset_decipher.csv"
df = load_data(DATA_PATH)


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_features(df):
    d = df.copy()

    # Transaction intensity proxy
    d["total_txn_count"] = (
        d["deposits"]
        + d["withdrawal"]
        + d["purchases"]
        + d["purchases_partners"]
    )

    d["total_purchase_count"] = d["purchases"] + d["purchases_partners"]

    # Rewards
    d["rewards_earned"] = d["rewards_earned"].fillna(0)
    d["reward_rate"] = d["reward_rate"].fillna(0)

    # Lending funnel proxy
    d["loan_interest_flag"] = (
        d["waiting_for_loan"]
        + d["cancelled_loan"]
        + d["received_loan"]
        + d["rejected_loan"]
    )

    # CC funnel proxy
    d["cc_interest_flag"] = (
        d["cc_application_begin"]
        + d["cc_liked"]
        + d["cc_disliked"]
        + d["cc_recommended"]
    )

    # Platform
    d["platform"] = np.select(
        [
            d.get("ios_user", 0) == 1,
            d.get("android_user", 0) == 1,
            d.get("web_user", 0) == 1,
            d.get("app_web_user", 0) == 1,
        ],
        ["ios", "android", "web", "app+web"],
        default="unknown",
    )

    # High value: top 10% by total transaction count
    d["is_high_value"] = (
        d["total_txn_count"] >= d["total_txn_count"].quantile(0.90)
    ).astype(int)

    # Churn label ensure int
    d["churn"] = d["churn"].astype(int)

    return d


df_feat = build_features(df)


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

churn_filter = st.sidebar.selectbox("Churn", ["All", "Active (0)", "Churned (1)"])

payment_types = ["All"] + sorted(df_feat["payment_type"].dropna().unique().tolist())
payment_filter = st.sidebar.selectbox("Payment Type", payment_types)

housing_types = ["All"] + sorted(df_feat["housing"].dropna().unique().tolist())
housing_filter = st.sidebar.selectbox("Housing", housing_types)

high_value_only = st.sidebar.checkbox("High Value Users Only", value=False)


def apply_filters(d):
    x = d.copy()

    if churn_filter == "Active (0)":
        x = x[x["churn"] == 0]
    elif churn_filter == "Churned (1)":
        x = x[x["churn"] == 1]

    if payment_filter != "All":
        x = x[x["payment_type"] == payment_filter]

    if housing_filter != "All":
        x = x[x["housing"] == housing_filter]

    if high_value_only:
        x = x[x["is_high_value"] == 1]

    return x


dff = apply_filters(df_feat)


# -----------------------------
# KPI SECTION
# -----------------------------
st.title("Paytm – Churn, Retention & Financial Dashboard (Python)")

col1, col2, col3, col4, col5 = st.columns(5)

users = len(dff)
churn_rate = dff["churn"].mean() if users > 0 else 0
avg_txn = dff["total_txn_count"].mean() if users > 0 else 0
loan_adopt = dff["received_loan"].mean() if users > 0 else 0
cc_adopt = dff["cc_taken"].mean() if users > 0 else 0

col1.metric("Users", f"{users:,}")
col2.metric("Churn Rate", f"{churn_rate*100:.2f}%")
col3.metric("Avg Txn Count", f"{avg_txn:.2f}")
col4.metric("Loan Adoption", f"{loan_adopt*100:.2f}%")
col5.metric("Credit Card Taken", f"{cc_adopt*100:.2f}%")


# -----------------------------
# ROW 1: CHURN BREAKDOWN
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(
        dff,
        x="total_txn_count",
        color="churn",
        nbins=40,
        title="Transaction Activity vs Churn",
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(
        dff,
        x="churn",
        y="reward_rate",
        title="Reward Rate vs Churn",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ROW 2: PAYMENTS + PLATFORM
# -----------------------------
c3, c4 = st.columns(2)

with c3:
    tmp = dff.groupby("payment_type")["churn"].mean().reset_index()
    tmp = tmp.sort_values("churn", ascending=False)
    fig = px.bar(tmp, x="payment_type", y="churn", title="Churn Rate by Payment Type")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    tmp = dff.groupby("platform")["churn"].mean().reset_index()
    tmp = tmp.sort_values("churn", ascending=False)
    fig = px.bar(tmp, x="platform", y="churn", title="Churn Rate by Platform")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ROW 3: LOANS + CREDIT CARD FUNNEL
# -----------------------------
st.subheader("Loan + Credit Card Funnel (Adoption Journey)")

c5, c6 = st.columns(2)

with c5:
    funnel_loan = pd.DataFrame({
        "stage": ["Waiting", "Rejected", "Cancelled", "Received"],
        "count": [
            int(dff["waiting_for_loan"].sum()),
            int(dff["rejected_loan"].sum()),
            int(dff["cancelled_loan"].sum()),
            int(dff["received_loan"].sum()),
        ]
    })
    fig = px.funnel(funnel_loan, x="count", y="stage", title="Loan Funnel")
    st.plotly_chart(fig, use_container_width=True)

with c6:
    funnel_cc = pd.DataFrame({
        "stage": ["Recommended", "Application Begun", "Liked", "Taken"],
        "count": [
            int(dff["cc_recommended"].sum()),
            int(dff["cc_application_begin"].sum()),
            int(dff["cc_liked"].sum()),
            int(dff["cc_taken"].sum()),
        ]
    })
    fig = px.funnel(funnel_cc, x="count", y="stage", title="Credit Card Funnel")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ROW 4: HIGH VALUE USERS
# -----------------------------
st.subheader("High Value Customers Retention")

hv = dff.copy()
tmp = hv.groupby("is_high_value")["churn"].mean().reset_index()
tmp["segment"] = tmp["is_high_value"].map({0: "Normal", 1: "High Value"})

fig = px.bar(tmp, x="segment", y="churn", title="Churn Rate: High Value vs Normal")
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# CHURN PREDICTION MODEL
# -----------------------------
st.subheader("Churn Prediction Model (Random Forest)")

model_df = df_feat.copy()

# Features for model
target = "churn"
drop_cols = ["user", "churn"]

X = model_df.drop(columns=drop_cols)
y = model_df[target]

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype != "object"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    max_depth=10,
    class_weight="balanced",
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

pred_proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_proba)

st.write(f"**ROC-AUC:** `{auc:.4f}`")

pred = (pred_proba >= 0.5).astype(int)
st.text(classification_report(y_test, pred))


# -----------------------------
# TOP FEATURES (APPROX)
# -----------------------------
# Extract feature importance from RF
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([cat_feature_names, np.array(num_cols)])

importances = pipe.named_steps["clf"].feature_importances_
fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values("importance", ascending=False).head(20)

fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Top 20 Churn Predictors")
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# USER RISK SCORING
# -----------------------------
st.subheader("Top Users at Risk of Churn (Risk Scoring)")

risk_df = df_feat.copy()
risk_X = risk_df.drop(columns=["user", "churn"])
risk_df["churn_risk_score"] = pipe.predict_proba(risk_X)[:, 1]

top_risk = risk_df.sort_values("churn_risk_score", ascending=False).head(25)[
    ["user", "churn_risk_score", "total_txn_count", "reward_rate", "received_loan", "cc_taken", "payment_type", "platform"]
]

st.dataframe(top_risk, use_container_width=True)

