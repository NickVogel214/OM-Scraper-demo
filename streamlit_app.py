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

def parse_om(uploaded_pdf) -> Dict[str, Any]:
    """
    Puppet OM parser.
    For this demo, we don't parse the PDF; we return a sample OM.
    Upload prompt is PDF to match your workflow/UX; swap in a real parser later.
    """
    # --- Sample OM values (rents intentionally HIGH vs Realtor for the demo) ---
    return {
        "address": "123 Main St, Tampa, FL",
        # Unit Info (rents high to show deviation)
        "avg_rent_1bd": 1650,  # realtor 1525
        "units_1bd": 31,
        "avg_rent_2bd": 2100,  # realtor 1935
        "units_2bd": 52,
        "avg_rent_3bd": 2450,  # realtor 2260
        "units_3bd": 29,
        "avg_rent_4bd": 2800,  # realtor 2550
        "units_4bd": 8,
        "avg_sqft_per_type": 920,

        # Location Data
        "lot_size": 1.85,
        "year_built_or_renov": 2001,
        "rentable_sqft": 108_750,
        "oz_status": "No",
        "total_units": 120,

        # Financials
        "noi": 1_890_000,
        "cap_rate": 0.054,
        "asking_price": 34_250_000,
        "expense_ratio": 0.41,
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    # Simulated market values
    return {
        "address": query or "123 Main St, Tampa, FL",
        # We include rents for value display; deviation uses OM vs Crexi
        "avg_rent_1bd": 1545,
        "avg_rent_2bd": 1970,
        "avg_rent_3bd": 2280,
        "avg_rent_4bd": 2580,
        "units_1bd": 32,
        "units_2bd": 48,
        "units_3bd": 30,
        "units_4bd": 10,
        "avg_sqft_per_type": 910,

        "lot_size": 1.70,
        "year_built_or_renov": 1999,
        "rentable_sqft": 107_200,
        "oz_status": "No",
        "total_units": 120,

        "noi": 1_825_000,
        "cap_rate": 0.056,
        "asking_price": 33_000_000,
        "expense_ratio": 0.40,
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
        "units_1bd": 29,           # sometimes inferred; may be missing in real life
        "units_2bd": 49,
        "total_units": 118,
        "asking_price": 33_500_000,
        # realtor often lacks many ops metrics; that's fine
    }

# ======================
# Metric schema + groups
# ======================
# rule: type + tolerance (either tol_abs or tol_rel) for color thresholds
METRICS: List[Tuple[str, Dict[str, Any]]] = [
    # Unit Info
    ("avg_rent_1bd", {"label": "Avg. Rent (1 Bed)", "group": "Unit Info", "type": "num", "tol_abs": 50}),
    ("units_1bd",    {"label": "# Units (1 Bed)",   "group": "Unit Info", "type": "num", "tol_abs": 2}),
    ("avg_rent_2bd", {"label": "Avg. Rent (2 Bed)", "group": "Unit Info", "type": "num", "tol_abs": 75}),
    ("units_2bd",    {"label": "# Units (2 Bed)",   "group": "Unit Info", "type": "num", "tol_abs": 2}),
    ("avg_rent_3bd", {"label": "Avg. Rent (3 Bed)", "group": "Unit Info", "type": "num", "tol_abs": 100}),
    ("units_3bd",    {"label": "# Units (3 Bed)",   "group": "Unit Info", "type": "num", "tol_abs": 2}),
    ("avg_rent_4bd", {"label": "Avg. Rent (4 Bed)", "group": "Unit Info", "type": "num", "tol_abs": 120}),
    ("units_4bd",    {"label": "# Units (4 Bed)",   "group": "Unit Info", "type": "num", "tol_abs": 2}),
    ("avg_sqft_per_type", {"label": "Avg. Sq. Ft. (per unit type)", "group": "Unit Info", "type": "num", "tol_abs": 50}),

    # Location Data
    ("address",            {"label": "Address", "group": "Location Data", "type": "str"}),
    ("lot_size",           {"label": "Lot Size (acres)", "group": "Location Data", "type": "num", "tol_rel": 0.10}),
    ("year_built_or_renov",{"label": "Year Built / Renovated", "group": "Location Data", "type": "num", "tol_abs": 3}),
    ("rentable_sqft",      {"label": "Rentable Sq. Ft.", "group": "Location Data", "type": "num", "tol_rel": 0.05}),
    ("oz_status",          {"label": "Opportunity Zone (OZ) Status", "group": "Location Data", "type": "str"}),
    ("total_units",        {"label": "Total Units", "group": "Location Data", "type": "num", "tol_abs": 2}),

    # Financials
    ("noi",            {"label": "NOI", "group": "Financials", "type": "num", "tol_rel": 0.05}),
    ("cap_rate",       {"label": "Cap Rate", "group": "Financials", "type": "num", "tol_rel": 0.01}),
    ("asking_price",   {"label": "Asking Price", "group": "Financials", "type": "num", "tol_rel": 0.03}),
    ("expense_ratio",  {"label": "Expense Ratio / Cost", "group": "Financials", "type": "num", "tol_rel": 0.05}),
]

GROUPS = ["Unit Info", "Location Data", "Financials"]
# Metrics we shouldn't comma-format (e.g., years)
NO_COMMA_KEYS = {"year_built_or_renov"}

# ======================
# Deviation math (OM vs Realtor, OM vs Crexi)
# ======================
def pct_dev_from_ref(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    if value is None or ref is None:
        return None
    if not isinstance(value, (int, float)) or not isinstance(ref, (int, float)):
        return None
    if ref == 0:
        return None
    return (float(value) - float(ref)) / float(ref)

def threshold_pct_vs_ref(rule: Dict[str, Any], ref: Optional[float]) -> float:
    """Convert tolerance into a percent-of-reference threshold."""
    if not isinstance(ref, (int, float)) or ref == 0:
        return 0.05  # fallback
    tol_rel = rule.get("tol_rel", None)
    if isinstance(tol_rel, (int, float)):
        return float(tol_rel)
    tol_abs = rule.get("tol_abs", None)
    if isinstance(tol_abs, (int, float)):
        return abs(float(tol_abs)) / abs(float(ref))
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

# ======================
# Table build + styling
# ======================
def build_rows(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any], group: str) -> pd.DataFrame:
    rows = []
    for key, rule in METRICS:
        if rule["group"] != group:
            continue
        label = rule.get("label", key)

        o = om.get(key)
        c = crexi.get(key)
        r = realtor.get(key)

        row = {
            "Metric": label,
            "OM": o,
            "Crexi": c,
            "Realtor": r,
            "OM vs Realtor Δ%": None,
            "OM vs Crexi Δ%": None,
            "_key": key,
            "_rule": rule,
        }

        if rule["type"] == "num":
            row["OM vs Realtor Δ%"] = pct_dev_from_ref(o, r)
            row["OM vs Crexi Δ%"]   = pct_dev_from_ref(o, c)

        rows.append(row)
    return pd.DataFrame(rows)

def fmt_value(key: str, x):
    if not isinstance(x, (int, float)):
        return x
    # years (and any keys in NO_COMMA_KEYS) -> no commas
    if key in NO_COMMA_KEYS:
        return f"{int(x)}" if float(x).is_integer() else f"{x}"
    # everything else: commas for thousands
    if abs(x) >= 1000:
        # keep decimals if present
        if float(x).is_integer():
            return f"{int(x):,d}"
        return f"{x:,.2f}"
    # small numbers: print raw or up to 3 decimals
    return f"{x:.3f}".rstrip("0").rstrip(".")

def fmt_pct(x):
    if isinstance(x, (int, float)):
        return f"{x*100:.1f}%"
    return ""

def style_group_df(df_raw: pd.DataFrame) -> "pd.io.formats.style.Styler":
    # Presentable copy with formatted values
    show = df_raw.copy()
    for i, row in df_raw.iterrows():
        key = row["_key"]
        for col in ["OM", "Crexi", "Realtor"]:
            show.at[i, col] = fmt_value(key, row[col])
        # format the deltas
        for col in ["OM vs Realtor Δ%", "OM vs Crexi Δ%"]:
            show.at[i, col] = fmt_pct(row[col])

    styler = show.drop(columns=["_key", "_rule"]).style

    # Color the deviation columns vs each source using that source as the tolerance base
    def colorize(colname: str, ref_col: str):
        colors = []
        for _, row in df_raw.iterrows():
            rule = row["_rule"]
            if rule["type"] != "num":
                colors.append("")
                continue
            ref_val = row[ref_col]  # Realtor or Crexi numeric value
            t = threshold_pct_vs_ref(rule, ref_val)
            dev = row[colname]
            colors.append(color_by_dev(dev, t))
        return colors

    styler = styler.apply(lambda _: colorize("OM vs Realtor Δ%", "Realtor"),
                          axis=0, subset=["OM vs Realtor Δ%"])
    styler = styler.apply(lambda _: colorize("OM vs Crexi Δ%", "Crexi"),
                          axis=0, subset=["OM vs Crexi Δ%"])

    # Light zebra-striping for readability
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# UI
# ======================
st.title("🏗️ CRE Benchmarking Demo (Puppet)")
st.caption("Shows **OM deviations** vs each market source (Realtor & Crexi). Grouped tables. PDF upload prompt (puppet).")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded_pdf = st.file_uploader("Drag & drop the OM **PDF** (optional)", type=["pdf"])
    st.markdown("**Note:** This demo doesn't parse the PDF; it uses sample OM values so you can show the flow.")
    st.divider()
    st.subheader("Adjust tolerances")
    # live tuning — keep rule keys clean (no tol_rel=None)
    for i, (key, rule) in enumerate(METRICS):
        if rule["type"] != "num":
            continue
        if "tol_abs" in rule:
            new_val = st.number_input(f"{rule['label']} tol_abs",
                                      value=float(rule["tol_abs"]), step=1.0, key=f"tol_abs_{key}")
            new_rule = {**rule, "tol_abs": new_val}
            if new_rule.get("tol_rel", None) is None:
                new_rule.pop("tol_rel", None)
            METRICS[i] = (key, new_rule)
        elif "tol_rel" in rule:
            new_val = st.number_input(f"{rule['label']} tol_rel",
                                      value=float(rule["tol_rel"]), step=0.005, format="%.3f", key=f"tol_rel_{key}")
            METRICS[i] = (key, {**rule, "tol_rel": new_val})

    run = st.button("Run Demo")

with st.expander("What this puppet does"):
    st.write("""
    - **OM vs Realtor Δ%** and **OM vs Crexi Δ%** only (no market averages).
    - **Color-coded** against each metric's tolerance (threshold scaled to the reference value).
    - **Grouped** output tables: Unit Info, Location Data, Financials — like your screenshot.
    - **OM rents are higher** than Realtor in the sample data to show noticeable deviations.
    - **PDF** drag-and-drop prompt (no parsing in demo; plug in your parser later).
    """)

if run:
    om = parse_om(uploaded_pdf)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    # GROUP 1: Unit Info
    st.subheader("Unit Info")
    df_unit = build_rows(om, crexi, realtor, "Unit Info")
    st.dataframe(style_group_df(df_unit), use_container_width=True, hide_index=True)

    # GROUP 2: Location Data
    st.subheader("Location Data")
    df_loc = build_rows(om, crexi, realtor, "Location Data")
    st.dataframe(style_group_df(df_loc), use_container_width=True, hide_index=True)

    # GROUP 3: Financials
    st.subheader("Financials")
    df_fin = build_rows(om, crexi, realtor, "Financials")
    st.dataframe(style_group_df(df_fin), use_container_width=True, hide_index=True)

    # Optional quick KPI: count how many OM metrics are within tolerance of each source
    def kpis_against(source_name: str, ref_col: str, df_all: pd.DataFrame):
        mask_num = df_all["_rule"].apply(lambda r: r["type"] == "num")
        subset = df_all[mask_num]
        within, total = 0, 0
        for _, row in subset.iterrows():
            dev = pct_dev_from_ref(row["OM"], row[ref_col])
            t = threshold_pct_vs_ref(row["_rule"], row[ref_col])
            if dev is not None:
                total += 1
                if abs(dev) <= t:
                    within += 1
        return within, total

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        w, t = kpis_against("Realtor", "Realtor", pd.concat([df_unit, df_loc, df_fin], ignore_index=True))
        st.metric("OM within tolerance vs Realtor", f"{w}/{t}")
    with c2:
        w, t = kpis_against("Crexi", "Crexi", pd.concat([df_unit, df_loc, df_fin], ignore_index=True))
        st.metric("OM within tolerance vs Crexi", f"{w}/{t}")
else:
    st.info("Drag & drop an OM **PDF** (optional), set a query, adjust tolerances, then click **Run Demo**.")
