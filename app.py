import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import json
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1️⃣ Smart ML Classification Engine
# ---------------------------------------------------
class LogClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

        self.patterns = {
            "Timeout/Latency": r"(longer than expected|duration=|timeout|latency|delayed)",
            "Resource Not Found": r"(404|not found|missing)",
            "Validation Error": r"(validation failed|status=400|invalid)",
            "Resource Conflict": r"(asset in use|409|conflict|already exists)",
            "Database Error": r"(db|connection|sql|query)",
            "Success": r"(successfully|ok|returned|health check)"
        }

    def _get_label_by_rule(self, msg):
        msg_l = msg.lower()
        for category, pattern in self.patterns.items():
            if re.search(pattern, msg_l):
                return category
        return "General System"

    def train_and_apply(self, df):
        if df.empty:
            return df

        labels = df['message'].apply(self._get_label_by_rule)
        X = self.vectorizer.fit_transform(df['message'])
        self.model.fit(X, labels)
        df['category'] = labels
        self.is_trained = True
        return df


@st.cache_resource
def get_classifier():
    return LogClassifier()

clf_tool = get_classifier()
# ---------------------------------------------------
# 2️⃣ Page Config & Auto Refresh
# ---------------------------------------------------
st.set_page_config(page_title="Log Pulse", layout="wide")
st_autorefresh(interval=10000, key="refresh")

# ---------------------------------------------------
# 3️⃣ Load Logs (JSON)
# ---------------------------------------------------
@st.cache_data
def load_data():
    with open('logs.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])

    def get_service(msg):
        match = re.search(r'\]\s+([\w\.]+)\s+:', msg)
        return match.group(1).split('.')[-1] if match else "System"

    df['service'] = df['message'].apply(get_service)
    df = clf_tool.train_and_apply(df)
    return df


df = load_data()

# ---------------------------------------------------
# 4️⃣ Alert Engine (UNCHANGED)
# ---------------------------------------------------
def get_alerts(df):
    alerts = []

    err_count = len(df[df['level'] == 'ERROR'])
    if err_count > 10:
        alerts.append({
            "name": "Critical Error Spike",
            "severity": "HIGH",
            "reason": f"System has {err_count} errors."
        })

    keyword_count = len(df[df['message'].str.contains("Asset in use", case=False)])
    if keyword_count > 5:
        alerts.append({
            "name": "Resource Contention",
            "severity": "LOW",
            "reason": f"'Asset in use' seen {keyword_count} times."
        })

    return alerts


# ---------------------------------------------------
# 5️⃣ UI Layout
# ---------------------------------------------------
st.title("Log Monitoring & Alerting")
st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')} | Smart Search Enabled")

# ---------------------------------------------------
# 6️⃣ Sidebar Filters (ONLY TIME FILTER ADDED)
# ---------------------------------------------------
st.sidebar.header("Controls")

levels = st.sidebar.multiselect(
    "Filter Levels",
    options=df['level'].unique(),
    default=df['level'].unique()
)

services = st.sidebar.multiselect(
    "Filter Services",
    options=df['service'].unique(),
    default=df['service'].unique()
)

search_input = st.sidebar.text_input(
    "Smart Search (try: timeout, validation, database)"
)

# 🕒 NEW: Time-based Filter (Last X Minutes)
time_window = st.sidebar.number_input(
    "Show logs from last X minutes",
    min_value=1,
    max_value=1440,
    value=5
)

# ---------------------------------------------------
# 7️⃣ Apply Filters
# ---------------------------------------------------
filtered_df = df[
    (df['level'].isin(levels)) &
    (df['service'].isin(services))
]

if search_input:
    # Split input by comma or space
    keywords = re.split(r'[,\s]+', search_input.lower())
    keywords = [k.strip() for k in keywords if k.strip()]

    combined_filter = pd.Series([False] * len(filtered_df))

    for keyword in keywords:
        # Match category
        cat_match = filtered_df['category'].str.contains(keyword, case=False)

        # Match message
        msg_match = filtered_df['message'].str.contains(keyword, case=False)

        combined_filter = combined_filter | cat_match | msg_match

    filtered_df = filtered_df[combined_filter]

# 🕒 Apply Time Window Filter
current_time = df['time'].max()
time_filtered_df = filtered_df[
    filtered_df['time'] >= current_time - pd.Timedelta(minutes=time_window)
]

# Export CSV
csv = time_filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Export CSV", csv, "logs.csv", "text/csv")

# ---------------------------------------------------
# 8️⃣ Alerts Display (UNCHANGED)
# ---------------------------------------------------
st.subheader("Active Alerts")
for a in get_alerts(df):
    with st.expander(f"{a['name']} - {a['severity']}", expanded=True):
        st.write(a['reason'])

# ---------------------------------------------------
# 9️⃣ Error Trend Chart (UNCHANGED)
# ---------------------------------------------------
st.subheader("Error count over time")
err_trend = df[df['level'] == 'ERROR'].resample('1min', on='time').count()['level']
st.line_chart(err_trend)

# ---------------------------------------------------
# 🔟 NEW: Pie Chart (Category-wise Errors in Time Window)
# ---------------------------------------------------
st.subheader(f"Error Distribution (Last {time_window} Minutes)")

error_only = time_filtered_df[time_filtered_df['level'] == 'ERROR']

if not error_only.empty:
    category_counts = error_only['category'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    ax.set_ylabel("")
    st.pyplot(fig)
else:
    st.info("No errors found in selected time window.")

# ---------------------------------------------------
# 1️⃣1️⃣ Styled Log Table (UNCHANGED except time filter applied)
# ---------------------------------------------------
def color_rows(row):
    if row.level == 'ERROR':
        return ['background-color: #f8d7da; color: black'] * len(row)
    if row.level == 'WARN':
        return ['background-color: #fff3cd; color: black'] * len(row)
    return [''] * len(row)


st.subheader("Filtered Logs (Time Window Applied)")
st.dataframe(
    time_filtered_df.sort_values(by='time', ascending=False)
    .style.apply(color_rows, axis=1),
    use_container_width=True
)

