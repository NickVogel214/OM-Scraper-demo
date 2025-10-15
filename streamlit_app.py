import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

st.set_page_config(page_title="CRE Scraping Demo (Puppet)", layout="wide")

# ======================
# Helpers / mock backends
# ======================
def try_float(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return x

def parse_om(uploaded_file) -> Dict[str, Any]:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            kv = {}
            for _, row in df.iterrows():
                kv[str(row["metric"]).strip()] = try_float(row["value"])
            return kv
        except Exception:
            st.warning("Couldn't parse the uploaded file; using sample OM instead.")

    return {
        "address": "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1550, "units_1bd": 30,
        "avg_rent_2bd": 1950, "units_2bd": 50,
        "avg_rent_3bd": 2300, "units_3bd": 30,
        "avg_rent_4bd": 2600, "units_4bd": 10,
        "avg_sqft_per_type": 900,
        "lot_size": 1.8, "year_built_or_renov": 2001,
        "rentable_sqft": 108000, "oz_status": "No",
        "total_units": 120,
        "noi": 1850000, "cap_rate": 0.055,
        "asking_price": 33600000, "expense_ratio": 0.38,
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
        "units_1bd": 32, "units_2bd": 48, "units_3bd": 30, "units_4bd": 10,
        "avg_sqft_per_type": 910, "lot_size": 1.7, "year_built_or_renov": 1999,
        "rentable_sqft": 107200, "oz_status": "No", "total_units": 120,
        "noi": 1825000, "cap_rate": 0.056, "asking_price": 33000000, "expense_ratio": 0.40,
        # we still include rents here for the value columns, but *rents' market avg* uses Realtor only
        "avg_rent_1bd": 1545, "avg_rent_2bd": 1970, "avg_rent_3bd": 2280, "avg_rent_4bd": 2580,
    }

def fetch_realtor(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1525, "avg_rent_2bd": 1935,
        "avg_rent_3bd": 2260, "avg_rent_4bd": 2550,
        "units_1bd": 29, "units_2bd": 49,
        "total_units": 118, "asking_price": 33500000,
    }

# ======================
# Metric schema
# ======================
METRICS: List[Tuple[str, Dict[str, Any]]] = [
    ("avg_rent_1bd", {"label": "Avg. Rent (1 Bed)", "type": "num", "tol_abs": 50}),
    ("units_1bd",    {"label": "# Units (1 Bed)",   "type": "num", "tol_abs": 2}),
    ("avg_rent_2bd", {"label": "Avg. Rent (2 Bed)", "type": "num", "tol_abs": 75}),
    ("units_2bd",    {"label": "# Units (2 Bed)",   "type": "num", "tol_abs": 2}),
    ("avg_rent_3bd", {"label": "Avg. Rent (3 Bed)", "type": "num", "tol_abs": 100}),
    ("units_3bd",    {"label": "# Units (3 Bed)",   "type": "num", "tol_abs": 2}),
    ("avg_rent_4bd", {"label": "Avg. Rent (4 Bed)", "type": "num", "tol_abs": 120}),
    ("units_4bd",    {"label": "# Units (4 Bed)",   "type": "num", "tol_abs": 2}),
    ("avg_sqft_per_type", {"label": "Avg. Sq. Ft. (per unit type)", "type": "num", "tol_abs": 50}),
    ("address",           {"label": "Address", "type": "str"}),
    ("lot_size",          {"label": "Lot Size (acres)", "type": "num", "tol_rel": 0.10}),
    ("year_built_or_renov", {"label": "Property Age / Year Renovated", "type": "num", "tol_abs": 3}),
    ("rentable_sqft",     {"label": "Rentable Sq. Ft.", "type": "num", "tol_rel": 0.05}),
    ("oz_status",         {"label": "Opportunity Zone (OZ) Status", "type": "str"}),
    ("total_units",       {"label": "Total Units", "type": "num", "tol_abs": 2}),
    ("noi",           {"label": "NOI", "type": "num", "tol_rel": 0.05}),
    ("cap_rate",      {"label": "Cap Rate", "type": "num", "tol_rel": 0.01}),
    ("asking_price",  {"label": "Asking Price", "type": "num", "tol_rel": 0.03}),
    ("expense_ratio", {"label": "Expense Ratio / Cost", "type": "num", "tol_rel": 0.05}),
]

RENT_KEYS = {"avg_rent_1bd", "avg_rent_2bd", "avg_rent_3bd", "avg_rent_4bd"}

# ======================
# Deviation math
# ======================
def market_average_for_key(key: str, rule: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> Optional[float]:
    """Rents → Realtor only; else combine Crexi + Realtor numeric values."""
    vals = []
    if key in RENT_KEYS:
        v = realtor.get(key)
        if isinstance(v, (int, float)): vals.append(float(v))
    else:
        for src in (crexi, realtor):
            v = src.get(key)
            if isinstance(v, (int, float)): vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals))

def pct_dev(value: Optional[float], avg: Optional[float]) -> Optional[float]:
    if value is None or avg is None:
        return None
    if not isinstance(value, (int, float)) or not isinstance(avg, (int, float)):
        return None
    if avg == 0:
        return None
    return (float(value) - float(avg)) / float(avg)

def threshold_pct(rule: Dict[str, Any], avg: Optional[float]) -> float:
    """Convert tolerance to a percentage threshold around the market average."""
    if avg is None or not isinstance(avg, (int, float)) or avg == 0:
        # fallback threshold if avg is missing/zero
        return 0.05
    if "tol_rel" in rule:
        return float(rule["tol_rel"])
    if "tol_abs" in rule:
        return abs(float(rule["tol_abs"])) / abs(float(avg))
    return 0.05

def color_by_dev(dev: Optional[float], thresh: float) -> str:
    """Green within threshold, amber within 2×, red beyond."""
    if dev is None:
        return ""
    adev = abs(dev)
    if adev <= thresh:
        return "background-color: #c8e6c9"  # green
    if adev <= 2 * thresh:
        return "background-color: #ffe0b2"  # amber
    return "background-color: #ffcdd2"      # red

def build_table(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rule in METRICS:
        label = rule.get("label", key)
        mkt_avg = market_average_for_key(key, rule, crexi, realtor) if rule["type"] == "num" else None
        row = {
            "Metric": label,
            "Market Avg": mkt_avg,
            "OM": om.get(key),
            "Crexi": crexi.get(key),
            "Realtor": realtor.get(key),
            "_key": key,
            "_rule": rule,
        }
        if rule["type"] == "num":
            row["OM Δ%"] = pct_dev(om.get(key), mkt_avg)
            row["Crexi Δ%"] = pct_dev(crexi.get(key), mkt_avg)
            row["Realtor Δ%"] = pct_dev(realtor.get(key), mkt_avg)
        else:
            row["OM Δ%"] = row["Crexi Δ%"] = row["Realtor Δ%"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def format_num(x):
    if isinstance(x, (int, float)):
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x}"
    return x

def format_pct(x):
    if isinstance(x, (int, float)):
        return f"{x*100:.1f}%"
    return ""

def style_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Show pretty numbers
    show = df.copy()
    show["Market Avg"] = show["Market Avg"].apply(format_num)
    for col in ["OM", "Crexi", "Realtor"]:
        show[col] = show[col].apply(format_num)
    for col in ["OM Δ%", "Crexi Δ%", "Realtor Δ%"]:
        show[col] = show[col].apply(format_pct)

    styler = show.drop(columns=["_key", "_rule"]).style

    # Build per-cell color map using original numeric df + thresholds
    def colorize(colname):
        colors = []
        for i, row in df.iterrows():
            rule = row["_rule"]
            avg = row["Market Avg"]
            t = threshold_pct(rule, avg) if rule["type"] == "num" else 0.0
            dev = row[colname]
            colors.append(color_by_dev(dev, t))
        return colors

    for col in ["OM Δ%", "Crexi Δ%", "Realtor Δ%"]:
        styler = styler.apply(lambda _: colorize(col), axis=0, subset=[col])
    return styler

# ======================
# UI
# ======================
st.title("🏗️ CRE Benchmarking Demo (Puppet)")
st.caption("Shows percent deviation from the calculated market average (rents use Realtor-only; others combine Crexi+Realtor). Cells color-coded vs tolerance.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded = st.file_uploader("Upload OM (CSV key/value: metric,value)", type=["csv"])
    st.markdown("**Tip:** No upload? We’ll use a built-in sample OM.")
    st.divider()
    st.subheader("Adjust tolerances")
    # live tuning
    for i, (key, rule) in enumerate(METRICS):
        if rule["type"] == "num":
            if "tol_abs" in rule:
                new_val = st.number_input(f"{rule['label']} tol_abs", value=float(rule["tol_abs"]),
                                          step=1.0, key=f"tol_abs_{key}")
                METRICS[i] = (key, {**rule, "tol_abs": new_val, "tol_rel": rule.get("tol_rel")})
            elif "tol_rel" in rule:
                new_val = st.number_input(f"{rule['label']} tol_rel", value=float(rule["tol_rel"]),
                                          step=0.005, format="%.3f", key=f"tol_rel_{key}")
                METRICS[i] = (key, {**rule, "tol_rel": new_val})
    run = st.button("Run Demo")

with st.expander("What this puppet does"):
    st.write("""
    - Computes a **market average** per numeric metric:
        - **Rents** → average of **Realtor only**.
        - **All other metrics** → average of **Crexi + Realtor** (available values).
    - Displays **percent deviation** from that average for **OM, Crexi, Realtor**.
    - **Color codes** deviations relative to the metric's tolerance:
        - Green: within tolerance
        - Amber: within 2× tolerance
        - Red: beyond 2× tolerance
    - Swap out the mock fetch/parse functions for production.
    """)

if run:
    om = parse_om(uploaded)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    df = build_table(om, crexi, realtor)

    st.subheader("Deviation vs Market Average")
    st.dataframe(
        style_table(df),
        use_container_width=True,
        hide_index=True
    )

    # Quick KPIs
    numeric_mask = df["_rule"].apply(lambda r: r["type"] == "num")
    kpi_df = df[numeric_mask]
    om_within = 0
    total = 0
    for _, row in kpi_df.iterrows():
        t = threshold_pct(row["_rule"], row["Market Avg"])
        dev = row["OM Δ%"]
        if dev is not None:
            total += 1
            if abs(dev) <= t:
                om_within += 1
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Numeric Metrics Evaluated", total)
    with col2:
        st.metric("OM within tolerance", f"{om_within}/{total}")
else:
    st.info("Upload an OM (optional), set a query, tweak tolerances, then click **Run Demo**.")

st.markdown("---")
st.caption("Puppet build for demo purposes. Replace the backends with real scrapers for production.")
