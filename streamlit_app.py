import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

st.set_page_config(page_title="CRE Scraping Demo (Puppet)", layout="wide")

# -----------------------
# Mock backends (puppet)
# -----------------------
def try_float(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return x

def parse_om(uploaded_file) -> Dict[str, Any]:
    """
    Puppet OM parser.
    - If a CSV is uploaded, we'll read key/value pairs: metric,value
    - Otherwise, return canned sample values.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Expect columns: metric, value
            kv = {}
            for _, row in df.iterrows():
                kv[str(row["metric"]).strip()] = try_float(row["value"])
            return kv
        except Exception:
            st.warning("Couldn't parse the uploaded file; using sample OM values instead.")
    # Sample demo values (replace later with real parser)
    return {
        "address": "123 Main St, Tampa, FL",
        "units_total": 120,
        "avg_rent_1bd": 1550,
        "avg_rent_2bd": 1950,
        "noi": 1850000,
        "cap_rate": 0.055,
        "asking_price": 33600000,
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    # Simulated based on query to keep it deterministic for a demo
    return {
        "address": query or "123 Main St, Tampa, FL",
        "units_total": 120,
        "avg_rent_1bd": 1545,
        "avg_rent_2bd": 1970,
        "noi": 1825000,
        "cap_rate": 0.056,
        "asking_price": 33000000,
    }

def fetch_realtor(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
        "units_total": 118,
        "avg_rent_1bd": 1525,
        "avg_rent_2bd": 1935,
        "asking_price": 33500000,
        # realtor typically lacks NOI, cap_rate
    }

# Which metrics to compare + their tolerance rules
METRICS = [
    ("address",       {"type": "str"}),
    ("units_total",   {"type": "num", "tol_abs": 2}),
    ("avg_rent_1bd",  {"type": "num", "tol_abs": 50}),
    ("avg_rent_2bd",  {"type": "num", "tol_abs": 75}),
    ("noi",           {"type": "num", "tol_rel": 0.05}),
    ("cap_rate",      {"type": "num", "tol_rel": 0.01}),
    ("asking_price",  {"type": "num", "tol_rel": 0.03}),
]

def format_num(x):
    if isinstance(x, (int, float)):
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x}"
    return x

def within_tolerance(a, b, rule: Dict[str, Any]) -> Optional[bool]:
    if a is None or b is None:
        return None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if "tol_abs" in rule:
        return abs(a - b) <= rule["tol_abs"]
    if "tol_rel" in rule:
        if b == 0:
            return a == 0
        return abs(a - b) <= abs(b) * rule["tol_rel"]
    return None

def build_compare_table(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rule in METRICS:
        o = om.get(key)
        c = crexi.get(key)
        r = realtor.get(key)
        comp_c = within_tolerance(o, c, rule) if rule["type"] == "num" else (str(o) == str(c) if (o is not None and c is not None) else None)
        comp_r = within_tolerance(o, r, rule) if rule["type"] == "num" else (str(o) == str(r) if (o is not None and r is not None) else None)
        rows.append({
            "metric": key,
            "OM": o,
            "Crexi": c,
            "Realtor": r,
            "OM≈Crexi": comp_c,
            "OM≈Realtor": comp_r,
        })
    df = pd.DataFrame(rows)
    for col in ["OM", "Crexi", "Realtor"]:
        df[col] = df[col].apply(format_num)
    return df

def color_match(val):
    if val is True:
        return "background-color: #c8e6c9"  # green-ish
    if val is False:
        return "background-color: #ffcdd2"  # red-ish
    return ""

# -----------------------
# UI
# -----------------------
st.title("🏗️ CRE Benchmarking Demo (Puppet)")
st.caption("Upload an OM CSV (optional), type a query, and see OM / Crexi / Realtor compared with tolerance flags.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded = st.file_uploader("Upload OM (CSV key/value: metric,value)", type=["csv"])
    st.markdown("**Tip:** No upload? We’ll use a built-in sample OM.")
    st.divider()
    st.subheader("Tolerance settings")
    for i, (metric, rule) in enumerate(METRICS):
        if rule["type"] == "num":
            if "tol_abs" in rule:
                new_val = st.number_input(f"{metric} tol_abs", value=float(rule["tol_abs"]), step=1.0, key=f"tol_abs_{metric}")
                METRICS[i] = (metric, {"type": "num", "tol_abs": new_val})
            elif "tol_rel" in rule:
                new_val = st.number_input(f"{metric} tol_rel", value=float(rule["tol_rel"]), step=0.005, format="%.3f", key=f"tol_rel_{metric}")
                METRICS[i] = (metric, {"type": "num", "tol_rel": new_val})
    run = st.button("Run Demo")

with st.expander("What this Demo does"):
    st.write("""
    - Builds a **3-column** table: OM / Crexi / Realtor.
    - Adds **comparison flags** for OM≈Crexi and OM≈Realtor using tolerances.
    - Lets you **tune tolerances** live.
    """)

if run:
    om = parse_om(uploaded)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    df = build_compare_table(om, crexi, realtor)

    st.subheader("Results")
    st.dataframe(
        df.style.applymap(color_match, subset=["OM≈Crexi", "OM≈Realtor"]),
        use_container_width=True,
        hide_index=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Metrics Compared", len(df))
    with col2:
        st.metric("OM≈Crexi (count)", int((df["OM≈Crexi"] == True).sum()))
    with col3:
        st.metric("OM≈Realtor (count)", int((df["OM≈Realtor"] == True).sum()))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv, file_name="compare_results.csv", mime="text/csv")
else:
    st.info("Set a query, (optionally) upload an OM CSV, tweak tolerances, then click **Run Demo** in the sidebar.")

st.markdown("---")
st.caption("Puppet build for demo purposes. Swap the backends with real scrapers for production.")
