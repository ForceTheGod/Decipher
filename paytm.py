import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Paytm Retention & Growth Dashboard", layout="wide")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("698627992e03e_Round_2_dataset_decipher.csv")

    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["housing", "payment_type", "zodiac_sign"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "housing" in df.columns:
        df["housing"] = df["housing"].replace({"na": np.nan, "nan": np.nan})

    return df


df = load_data()


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_features(df):
    d = df.copy()

    d["total_txn_count"] = (
        d["deposits"]
        + d["withdrawal"]
        + d["purchases"]
        + d["purchases_partners"]
    )

    d["total_purchase_count"] = d["purchases"] + d["purchases_partners"]

    d["rewards_earned"] = d["rewards_earned"].fillna(0)
    d["reward_rate"] = d["reward_rate"].fillna(0)

    d["loan_interest_flag"] = (
        d["waiting_for_loan"]
        + d["cancelled_loan"]
        + d["received_loan"]
        + d["rejected_loan"]
    )

    d["cc_interest_flag"] = (
        d["cc_application_begin"]
        + d["cc_liked"]
        + d["cc_disliked"]
        + d["cc_recommended"]
    )

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

    # High value = top 10% by transaction count
    d["is_high_value"] = (
        d["total_txn_count"] >= d["total_txn_count"].quantile(0.90)
    ).astype(int)

    d["churn"] = d["churn"].astype(int)

    # engagement buckets
    d["txn_bucket"] = pd.cut(
        d["total_txn_count"],
        bins=[-1, 2, 8, 20, 999999],
        labels=["Low (0-2)", "Medium (3-8)", "High (9-20)", "Power (20+)"]
    )

    d["reward_bucket"] = pd.cut(
        d["reward_rate"],
        bins=[-0.01, 0.05, 0.2, 1.0],
        labels=["Weak", "Moderate", "Strong"]
    )

    return d


df_feat = build_features(df)


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

churn_filter = st.sidebar.selectbox("Churn", ["All", "Active (0)", "Churned (1)"])

payment_types = ["All"] + sorted(df_feat["payment_type"].dropna().unique().tolist())
payment_filter = st.sidebar.selectbox("Payment Type", payment_types)

platform_types = ["All"] + sorted(df_feat["platform"].dropna().unique().tolist())
platform_filter = st.sidebar.selectbox("Platform", platform_types)

high_value_only = st.sidebar.checkbox("High Value Users Only", value=False)


def apply_filters(d):
    x = d.copy()

    if churn_filter == "Active (0)":
        x = x[x["churn"] == 0]
    elif churn_filter == "Churned (1)":
        x = x[x["churn"] == 1]

    if payment_filter != "All":
        x = x[x["payment_type"] == payment_filter]

    if platform_filter != "All":
        x = x[x["platform"] == platform_filter]

    if high_value_only:
        x = x[x["is_high_value"] == 1]

    return x


dff = apply_filters(df_feat)


# -----------------------------
# KPI SECTION
# -----------------------------
st.title("Paytm – Retention, Engagement & Adoption Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)

users = len(dff)
churn_rate = dff["churn"].mean() if users > 0 else 0
avg_txn = dff["total_txn_count"].mean() if users > 0 else 0
loan_adopt = dff["received_loan"].mean() if users > 0 else 0
cc_adopt = dff["cc_taken"].mean() if users > 0 else 0
reward_avg = dff["reward_rate"].mean() if users > 0 else 0

col1.metric("Users", f"{users:,}")
col2.metric("Churn Rate", f"{churn_rate*100:.2f}%")
col3.metric("Avg Txn Count", f"{avg_txn:.2f}")
col4.metric("Loan Adoption", f"{loan_adopt*100:.2f}%")
col5.metric("Avg Reward Rate", f"{reward_avg*100:.2f}%")


# -----------------------------
# SECTION 1: CHURN BEHAVIOR
# -----------------------------
st.subheader("1) Why Users Churn (Behavior + Engagement)")

c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(
        dff,
        x="total_txn_count",
        color="churn",
        nbins=40,
        title="Transaction Activity vs Churn"
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(
        dff,
        x="churn",
        y="reward_rate",
        title="Reward Rate vs Churn"
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# SECTION 2: SEGMENTS TO TARGET
# -----------------------------
st.subheader("2) Segments With Highest Churn (Targeting Map)")

seg = (
    dff.groupby(["txn_bucket", "reward_bucket"])["churn"]
    .mean()
    .reset_index()
    .rename(columns={"churn": "churn_rate"})
)

fig = px.density_heatmap(
    seg,
    x="txn_bucket",
    y="reward_bucket",
    z="churn_rate",
    title="Churn Heatmap: Transaction Activity vs Reward Strength",
    text_auto=".2f"
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Action Insight: Focus retention on **Low/Medium transaction users** with **Weak rewards**. "
    "These segments typically have the highest churn and are easiest to reactivate."
)


# -----------------------------
# SECTION 3: LOAN + CC FUNNEL
# -----------------------------
st.subheader("3) Improve Loan & Credit Card Adoption")

c3, c4 = st.columns(2)

with c3:
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

with c4:
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

st.success(
    "Action Insight: The biggest drop-off is usually between **Recommendation → Application Begin** "
    "and **Liked → Taken**. Improve onboarding, reduce KYC friction, and offer clear reward-based reasons to convert."
)


# -----------------------------
# SECTION 4: HIGH VALUE RETENTION
# -----------------------------
st.subheader("4) Improve Retention of High-Value Customers")

tmp = df_feat.groupby("is_high_value")["churn"].mean().reset_index()
tmp["segment"] = tmp["is_high_value"].map({0: "Normal", 1: "High Value"})

fig = px.bar(tmp, x="segment", y="churn", title="Churn Rate: High Value vs Normal")
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "High-value churn is expensive. Treat these users like VIPs: dedicated offers, priority support, "
    "and personalized rewards based on spending behavior."
)


# -----------------------------
# SECTION 5: CHURN RISK SCORING (NO ROC-AUC)
# -----------------------------
st.subheader("5) Predict Users at Risk of Churn (Actionable Risk List)")

model_df = df_feat.drop(columns=["txn_bucket", "reward_bucket"], errors="ignore").copy()


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

# Train model (no evaluation shown)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

# Risk scoring for all users
risk_df = df_feat.copy()
risk_X = risk_df.drop(columns=["user", "churn"])
risk_df["churn_risk_score"] = pipe.predict_proba(risk_X)[:, 1]


# -----------------------------
# TOP DRIVERS (FEATURE IMPORTANCE)
# -----------------------------
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([cat_feature_names, np.array(num_cols)])

importances = pipe.named_steps["clf"].feature_importances_
fi = pd.DataFrame({"feature": feature_names, "importance": importances})
fi = fi.sort_values("importance", ascending=False).head(15)

c5, c6 = st.columns([1.1, 1])

with c5:
    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Reasons for Churn (Model Explainability)"
    )
    st.plotly_chart(fig, use_container_width=True)

with c6:
    st.markdown("### Recommended actions for retention")
    st.write("Use churn drivers to pick actions:")
    st.markdown("""
- **Low transactions** → notify constantly with cashback offers and UPI reminders  
- **Weak reward rate** → give personalized rewards and discounts  
- **Loan/CC drop-off** → simplify KYC and improve eligibility  
- **High-value churn** → VIP program and priority support
    """)


# -----------------------------
# TOP USERS AT RISK
# -----------------------------
st.subheader("Top Users Most Likely to Churn (Prioritize Retention)")

top_risk = risk_df.sort_values("churn_risk_score", ascending=False).head(25)[
    [
        "user", "churn_risk_score", "total_txn_count",
        "reward_rate", "received_loan", "cc_taken",
        "payment_type", "platform", "is_high_value"
    ]
]

st.dataframe(top_risk, use_container_width=True)


# -----------------------------
# SECTION 6: OPPORTUNITY DASHBOARD
# -----------------------------
st.subheader("6) Growth Opportunities (What To Improve)")

opp1 = ((df_feat["total_txn_count"] <= 2) & (df_feat["churn"] == 0)).mean()
opp2 = ((df_feat["reward_rate"] < 0.05) & (df_feat["churn"] == 0)).mean()
opp3 = ((df_feat["cc_recommended"] == 1) & (df_feat["cc_taken"] == 0)).mean()
opp4 = ((df_feat["waiting_for_loan"] == 1) & (df_feat["received_loan"] == 0)).mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Active users with low txn activity", f"{opp1*100:.1f}%")
k2.metric("Active users with weak rewards", f"{opp2*100:.1f}%")
k3.metric("Credit card recommended but not taken", f"{opp3*100:.1f}%")
k4.metric("Loan waiting but not received", f"{opp4*100:.1f}%")

st.info(
    "These are your best targets: users who are still active but show weak engagement. "
    "They are easier to convert than fully churned users."
)

