import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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
    """
    Puppet OM parser.
    If CSV uploaded, expects columns: metric,value
    Otherwise returns a sample OM.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            kv = {}
            for _, row in df.iterrows():
                kv[str(row["metric"]).strip()] = try_float(row["value"])
            return kv
        except Exception:
            st.warning("Couldn't parse the uploaded file; using sample OM instead.")

    # --- Sample OM values (edit as you like) ---
    return {
        "address": "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1550,
        "units_1bd": 30,
        "avg_rent_2bd": 1950,
        "units_2bd": 50,
        "avg_rent_3bd": 2300,
        "units_3bd": 30,
        "avg_rent_4bd": 2600,
        "units_4bd": 10,
        "avg_sqft_per_type": 900,
        "lot_size": 1.8,
        "year_built_or_renov": 2001,
        "rentable_sqft": 108000,
        "oz_status": "No",
        "total_units": 120,
        "noi": 1850000,
        "cap_rate": 0.055,
        "asking_price": 33600000,
        "expense_ratio": 0.38,
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    # Simulated market values
    return {
        "address": query or "123 Main St, Tampa, FL",
        "units_1bd": 32,
        "units_2bd": 48,
        "units_3bd": 30,
        "units_4bd": 10,
        "avg_sqft_per_type": 910,
        "lot_size": 1.7,
        "year_built_or_renov": 1999,
        "rentable_sqft": 107200,
        "oz_status": "No",
        "total_units": 120,
        "noi": 1825000,
        "cap_rate": 0.056,
        "asking_price": 33000000,
        "expense_ratio": 0.40,
        # note: we intentionally DO NOT provide rents here; rents come from Realtor only for plots
        "avg_rent_1bd": 1545,
        "avg_rent_2bd": 1970,
        "avg_rent_3bd": 2280,
        "avg_rent_4bd": 2580,
    }

def fetch_realtor(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1525,
        "avg_rent_2bd": 1935,
        "avg_rent_3bd": 2260,
        "avg_rent_4bd": 2550,
        "units_1bd": 29,  # sometimes inferred; ok if missing later
        "units_2bd": 49,
        "total_units": 118,
        "asking_price": 33500000,
        # realtor often lacks NOI/cap_rate/etc.; that's fine
    }

# ======================
# Metric schema (your table)
# ======================
# (name, rule) where rule defines type and tolerance model for comparisons & synthetic spread
METRICS: List[Tuple[str, Dict[str, Any]]] = [
    # Unit Information (rents use Realtor only for plots)
    ("avg_rent_1bd", {"label": "Avg. Rent (1 Bed)", "type": "num", "tol_abs": 50}),
    ("units_1bd",    {"label": "# Units (1 Bed)",  "type": "num", "tol_abs": 2}),
    ("avg_rent_2bd", {"label": "Avg. Rent (2 Bed)", "type": "num", "tol_abs": 75}),
    ("units_2bd",    {"label": "# Units (2 Bed)",  "type": "num", "tol_abs": 2}),
    ("avg_rent_3bd", {"label": "Avg. Rent (3 Bed)", "type": "num", "tol_abs": 100}),
    ("units_3bd",    {"label": "# Units (3 Bed)",  "type": "num", "tol_abs": 2}),
    ("avg_rent_4bd", {"label": "Avg. Rent (4 Bed)", "type": "num", "tol_abs": 120}),
    ("units_4bd",    {"label": "# Units (4 Bed)",  "type": "num", "tol_abs": 2}),
    ("avg_sqft_per_type", {"label": "Avg. Sq. Ft. (per unit type)", "type": "num", "tol_abs": 50}),

    # Location Data
    ("address",           {"label": "Address", "type": "str"}),
    ("lot_size",          {"label": "Lot Size (acres)", "type": "num", "tol_rel": 0.10}),
    ("year_built_or_renov", {"label": "Property Age / Year Renovated", "type": "num", "tol_abs": 3}),
    ("rentable_sqft",     {"label": "Rentable Sq. Ft.", "type": "num", "tol_rel": 0.05}),
    ("oz_status",         {"label": "Opportunity Zone (OZ) Status", "type": "str"}),
    ("total_units",       {"label": "Total Units", "type": "num", "tol_abs": 2}),

    # Financials
    ("noi",           {"label": "NOI", "type": "num", "tol_rel": 0.05}),
    ("cap_rate",      {"label": "Cap Rate", "type": "num", "tol_rel": 0.01}),
    ("asking_price",  {"label": "Asking Price", "type": "num", "tol_rel": 0.03}),
    ("expense_ratio", {"label": "Expense Ratio / Cost", "type": "num", "tol_rel": 0.05}),
]

RENT_KEYS = {"avg_rent_1bd", "avg_rent_2bd", "avg_rent_3bd", "avg_rent_4bd"}

# ======================
# Comparison & table build
# ======================
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
        return abs(a - b) <= float(rule["tol_abs"])
    if "tol_rel" in rule:
        if b == 0:
            return a == 0
        return abs(a - b) <= abs(b) * float(rule["tol_rel"])
    return None

def build_compare_table(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rule in METRICS:
        o = om.get(key)
        c = crexi.get(key)
        r = realtor.get(key)
        if rule["type"] == "num":
            comp_c = within_tolerance(o, c, rule)
            comp_r = within_tolerance(o, r, rule)
        else:
            comp_c = (str(o) == str(c)) if (o is not None and c is not None) else None
            comp_r = (str(o) == str(r)) if (o is not None and r is not None) else None

        rows.append({
            "Metric": rule.get("label", key),
            "OM": o,
            "Crexi": c,
            "Realtor": r,
            "OM≈Crexi": comp_c,
            "OM≈Realtor": comp_r,
            "_key": key,          # keep for plotting
            "_rule": rule,        # keep for plotting
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

# ======================
# Box plots (market-only combined; OM as line)
# ======================
def _std_from_rule(rule: Dict[str, Any], value: Optional[float]) -> Optional[float]:
    if value is None or not isinstance(value, (int, float)):
        return None
    if "tol_abs" in rule:
        return max(float(rule["tol_abs"]), 1e-9)
    if "tol_rel" in rule:
        return max(abs(float(value)) * float(rule["tol_rel"]), 1e-9)
    return None

def _synthetic(mu: float, sigma: float, n: int = 400) -> np.ndarray:
    sigma = max(sigma, 1e-9)
    # Using the same generator ensures "normalized" comparability across sources
    return np.random.normal(loc=mu, scale=sigma, size=n)

def combined_market_samples(key: str, rule: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> np.ndarray:
    """Combine market points into a single array; for rents, only use Realtor."""
    samples: List[np.ndarray] = []

    # Rents → Realtor only
    if key in RENT_KEYS:
        v = realtor.get(key)
        if isinstance(v, (int, float)):
            s = _std_from_rule(rule, v)
            if s is not None:
                samples.append(_synthetic(float(v), s))
    else:
        for src in (crexi, realtor):
            v = src.get(key)
            if isinstance(v, (int, float)):
                s = _std_from_rule(rule, v)
                if s is not None:
                    samples.append(_synthetic(float(v), s))

    if not samples:
        return np.array([])

    return np.concatenate(samples, axis=0)

def make_combined_boxplot(metric_label: str, key: str, rule: Dict[str, Any], om_v, crexi: Dict[str, Any], realtor: Dict[str, Any]):
    if rule.get("type") != "num":
        return None

    data = combined_market_samples(key, rule, crexi, realtor)
    if data.size == 0:
        return None

    fig = plt.figure()
    plt.boxplot([data], labels=["Market"], showmeans=True)

    if isinstance(om_v, (int, float)):
        y = float(om_v)
        plt.axhline(y=y, linestyle="--", linewidth=1.5)
        plt.text(1.12, y, f"OM = {y:.4g}", va="center")

    subtitle = "Realtor-only" if key in RENT_KEYS else "Crexi + Realtor"
    plt.title(f"{metric_label} — Market Box ({subtitle}); OM as dashed line")
    plt.ylabel(metric_label)
    plt.tight_layout()
    return fig

# ======================
# UI
# ======================
st.title("🏗️ CRE Benchmarking Demo (Puppet)")
st.caption("Market box plots combine all market points per metric (rents use Realtor only). OM is overlaid as a dashed line.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded = st.file_uploader("Upload OM (CSV key/value: metric,value)", type=["csv"])
    st.markdown("**Tip:** No upload? We’ll use a built-in sample OM.")
    st.divider()
    st.subheader("Tolerance settings")
    # live-tunable tolerances
    for i, (key, rule) in enumerate(METRICS):
        if rule["type"] == "num":
            if "tol_abs" in rule:
                new_val = st.number_input(f"{rule['label']} tol_abs", value=float(rule["tol_abs"]),
                                          step=1.0, key=f"tol_abs_{key}")
                METRICS[i] = (key, {**rule, "tol_abs": new_val, "tol_rel": rule.get("tol_rel", None)})
            elif "tol_rel" in rule:
                new_val = st.number_input(f"{rule['label']} tol_rel", value=float(rule["tol_rel"]),
                                          step=0.005, format="%.3f", key=f"tol_rel_{key}")
                METRICS[i] = (key, {**rule, "tol_rel": new_val})
    run = st.button("Run Demo")

with st.expander("What this puppet does"):
    st.write("""
    - **3-column** compare table: OM / Crexi / Realtor + match flags.
    - **One market box plot per metric**, placed next to the table:
        - **Combine** Crexi + Realtor samples for each numeric metric.
        - **Rents use only Realtor** for the market box.
        - **OM** is shown as a **dashed horizontal line** (no OM box).
    - Spreads are synthesized from the tolerance rules (abs or value×relative).
    - Replace `parse_om`, `fetch_crexi`, `fetch_realtor` with real code later.
    """)

if run:
    om = parse_om(uploaded)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Results")
        df = build_compare_table(om, crexi, realtor)
        # Store raw keys/rules for plotting, but hide them in the UI
        styled = df.drop(columns=["_key", "_rule"]).style.applymap(color_match, subset=["OM≈Crexi", "OM≈Realtor"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Metrics Compared", len(df))
        with c2:
            st.metric("OM≈Crexi (count)", int((df["OM≈Crexi"] == True).sum()))
        with c3:
            st.metric("OM≈Realtor (count)", int((df["OM≈Realtor"] == True).sum()))

        csv = df.drop(columns=["_key", "_rule"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", data=csv, file_name="compare_results.csv", mime="text/csv")

    with right:
        st.subheader("Market Box Plots")
        for _, row in df.iterrows():
            key = row["_key"]
            rule = row["_rule"]
            if rule.get("type") != "num":
                continue
            fig = make_combined_boxplot(row["Metric"], key, rule, om.get(key), crexi, realtor)
            if fig is not None:
                st.pyplot(fig)
else:
    st.info("Set a query, (optionally) upload an OM CSV, tweak tolerances, then click **Run Demo** in the sidebar.")

st.markdown("---")
st.caption("Puppet build for demo purposes. Swap the backends with real scrapers for production.")
