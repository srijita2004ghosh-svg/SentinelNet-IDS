# SentinelNet IDS – Enterprise AI SOC Platform
# AI + Risk Scoring + Blockchain Audit + Threat Intelligence
# Author: Samyak Biswas

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import hashlib

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="SentinelNet IDS | Enterprise SOC",
    page_icon="🛡️",
    layout="wide"
)

# ================== THREAT INTELLIGENCE DATABASE ==================
THREAT_DB = {
    "NORMAL": {
        "level": "LOW",
        "base_risk": 10,
        "description": "Legitimate network traffic.",
        "impact": "No security impact.",
        "action": "Allow traffic"
    },
    "PROBE": {
        "level": "MEDIUM",
        "base_risk": 40,
        "description": "Scanning network for open ports and vulnerabilities.",
        "impact": "Reconnaissance before an attack.",
        "action": "Monitor, rate-limit source"
    },
    "DOS": {
        "level": "HIGH",
        "base_risk": 70,
        "description": "Denial of Service attack from a single source.",
        "impact": "Service slowdown or outage.",
        "action": "Block IP, apply traffic filtering"
    },
    "DDOS": {
        "level": "CRITICAL",
        "base_risk": 90,
        "description": "Distributed DoS from multiple compromised systems.",
        "impact": "Complete service disruption.",
        "action": "Activate DDoS mitigation & SOC escalation"
    },
    "R2L": {
        "level": "HIGH",
        "base_risk": 75,
        "description": "Remote attacker attempting local access.",
        "impact": "Credential compromise.",
        "action": "Enforce authentication controls"
    },
    "U2R": {
        "level": "CRITICAL",
        "base_risk": 95,
        "description": "Local privilege escalation attack.",
        "impact": "Full system compromise.",
        "action": "Immediate isolation & forensics"
    }
}

ATTACK_TYPES = list(THREAT_DB.keys())

# ================== SIDEBAR ==================
st.sidebar.title("🛡️ SentinelNet IDS")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
model_choice = st.sidebar.radio(
    "Detection Model",
    ["Isolation Forest (Unsupervised)", "Random Forest (Supervised)"]
)
st.sidebar.markdown("---")
st.sidebar.info("AI SOC • Risk Engine • Blockchain Audit")

# ================== HEADER ==================
st.title("🛡️ SentinelNet – AI Intrusion Detection System")
st.caption("Threat Intelligence • Risk Scoring • Blockchain-backed SOC")

# ================== DATA ==================
def generate_data():
    records = 9000

    timestamps = pd.date_range(
        start=datetime.now(),
        periods=records,
        freq="s"
    )

    return pd.DataFrame({
        "duration": np.random.randint(0, 1000, records),
        "src_bytes": np.random.randint(0, 100000, records),
        "dst_bytes": np.random.randint(0, 100000, records),
        "failed_logins": np.random.randint(0, 5, records),
        "attack_type": np.random.choice(
            ATTACK_TYPES,
            records,
            p=[0.20, 0.10, 0.20, 0.05, 0.15, 0.30]
        ),
        "timestamp": timestamps
    })

df = pd.read_csv(uploaded_file) if uploaded_file else generate_data()

# ================== FEATURES ==================
features = ["duration", "src_bytes", "dst_bytes", "failed_logins"]
X = df[features]
y_binary = np.where(df["attack_type"] == "NORMAL", 0, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== ML MODEL ==================
if "Isolation Forest" in model_choice:
    model = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
    df["ml_attack"] = np.where(model.fit_predict(X_scaled) == -1, 1, 0)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.3, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    df["ml_attack"] = model.predict(X_scaled)

# ================== RISK SCORE ==================
def calc_risk(row):
    base = THREAT_DB[row["attack_type"]]["base_risk"]
    anomaly_bonus = 15 if row["ml_attack"] == 1 else 0
    login_bonus = min(row["failed_logins"] * 5, 15)
    return min(base + anomaly_bonus + login_bonus, 100)

df["risk_score"] = df.apply(calc_risk, axis=1)
df["severity"] = np.where(df["risk_score"] >= 70, "HIGH", "NORMAL")

# ================== BLOCKCHAIN AUDIT ==================
def hash_block(i, ts, data, prev):
    return hashlib.sha256(f"{i}{ts}{data}{prev}".encode()).hexdigest()

blocks = []
prev_hash = "0"

for _, r in df[df["severity"] == "HIGH"].iterrows():
    block = {
        "index": len(blocks),
        "timestamp": str(r["timestamp"]),
        "attack": r["attack_type"],
        "risk": r["risk_score"],
        "prev_hash": prev_hash
    }
    block["hash"] = hash_block(
        block["index"], block["timestamp"],
        block["attack"] + str(block["risk"]), prev_hash
    )
    prev_hash = block["hash"]
    blocks.append(block)

audit_df = pd.DataFrame(blocks)

# ================== METRICS ==================
st.subheader("📊 SOC Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Events", len(df))
m2.metric("High-Risk Events", (df["risk_score"] >= 70).sum())
m3.metric("Average Risk", f"{df['risk_score'].mean():.1f}")
m4.metric("Blockchain Blocks", len(audit_df))

# ================== THREAT ANALYTICS ==================
st.subheader("📈 Threat Analytics")

c1, c2 = st.columns(2)

with c1:
    fig1 = px.bar(
        df["attack_type"].value_counts(),
        title="Threat Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.pie(
        df,
        names="attack_type",
        title="Attack Share",
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================== RISK VISUALS ==================
st.subheader("🔥 Risk Analysis")

r1, r2 = st.columns(2)

with r1:
    st.plotly_chart(
        px.histogram(df, x="risk_score", nbins=20, title="Risk Score Distribution"),
        use_container_width=True
    )

with r2:
    st.plotly_chart(
        px.box(df, x="attack_type", y="risk_score", title="Risk per Threat Type"),
        use_container_width=True
    )

# ================== TIMELINE ==================
timeline = (
    df.groupby(
        [pd.Grouper(key="timestamp", freq="30s"), "attack_type"]
    )
    .size()
    .reset_index(name="count")
)

st.plotly_chart(
    px.area(
        timeline,
        x="timestamp",
        y="count",
        color="attack_type",
        title="Threat Activity Timeline"
    ),
    use_container_width=True
)

# ================== THREAT INTELLIGENCE PANEL ==================
st.subheader("🧠 Threat Intelligence Panel")

selected = st.selectbox("Select Threat", ATTACK_TYPES)
info = THREAT_DB[selected]

a, b, c = st.columns(3)
a.metric("Threat Level", info["level"])
b.metric("Base Risk", info["base_risk"])
c.metric("Events Detected", (df["attack_type"] == selected).sum())

st.info(f"📌 **Description:** {info['description']}")
st.warning(f"⚠️ **Impact:** {info['impact']}")
st.success(f"✅ **SOC Action:** {info['action']}")

# ================== BLOCKCHAIN VIEW ==================
st.subheader("⛓️ Blockchain Audit Trail")
st.dataframe(audit_df.tail(15), use_container_width=True)

# ================== EXPORT ==================
st.subheader("⬇️ Export Reports")

st.download_button(
    "Download SOC CSV",
    df.to_csv(index=False),
    "sentinelnet_soc_report.csv"
)

st.download_button(
    "Download Blockchain Audit CSV",
    audit_df.to_csv(index=False),
    "sentinelnet_blockchain_audit.csv"
)

# ================== FOOTER ==================
st.markdown("---")
st.caption("🛡️ SentinelNet IDS | AI • Risk • Threat Intelligence • Blockchain")
