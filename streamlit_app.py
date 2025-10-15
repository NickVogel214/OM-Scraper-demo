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
    For this demo we don't parse the PDF; we return a sample OM with rents higher than Realtor.
    Swap with your real PDF parser later.
    """
    return {
        "address": "123 Main St, Tampa, FL",
        # Unit Info (rents high vs Realtor for demo)
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

        # Financials (summary-only)
        "noi": 1_890_000,
        "cap_rate": 0.054,
        "asking_price": 34_250_000,
        "expense_ratio": 0.41,
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
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
        "units_1bd": 29,
        "units_2bd": 49,
        "total_units": 118,
        "asking_price": 33_500_000,
    }

# ======================
# Metric schema + groups
# ======================
RENT_KEYS = ["1bd", "2bd", "3bd", "4bd"]
RENT_METRICS = {
    "1bd": {"rent_key": "avg_rent_1bd", "units_key": "units_1bd", "label": "1 Bed", "tol_abs": 50},
    "2bd": {"rent_key": "avg_rent_2bd", "units_key": "units_2bd", "label": "2 Bed", "tol_abs": 75},
    "3bd": {"rent_key": "avg_rent_3bd", "units_key": "units_3bd", "label": "3 Bed", "tol_abs": 100},
    "4bd": {"rent_key": "avg_rent_4bd", "units_key": "units_4bd", "label": "4 Bed", "tol_abs": 120},
}

# Location Data — only these are compared
LOC_METRICS: List[Tuple[str, Dict[str, Any]]] = [
    ("total_units",       {"label": "Total Units",              "type": "num", "tol_abs": 2}),
    ("lot_size",          {"label": "Lot Size (acres)",         "type": "num", "tol_rel": 0.10}),
    ("year_built_or_renov", {"label": "Year Built / Renovated", "type": "num", "tol_abs": 3}),
    ("rentable_sqft",     {"label": "Rentable Sq. Ft.",         "type": "num", "tol_rel": 0.05}),
]

# Fields shown only in Property Summary (no comparisons table)
SUMMARY_FIELDS = [
    ("address",       "Address",             "text"),
    ("oz_status",     "OZ Status",           "text"),
    ("avg_sqft_per_type", "Avg Sq Ft / Unit","num"),
    ("noi",           "NOI",                 "money"),
    ("cap_rate",      "Cap Rate",            "percent"),
    ("asking_price",  "Asking Price",        "money"),
    ("expense_ratio", "Expense Ratio",       "percent"),
]

NO_COMMA_KEYS = {"year_built_or_renov"}

# ======================
# Formatting + tolerance helpers
# ======================
def fmt_money(x):
    if isinstance(x, (int, float)):
        return f"${int(round(x)):,}"
    return x

def fmt_percent(x):
    if isinstance(x, (int, float)):
        return f"{x*100:.1f}%"
    return x

def fmt_number(key: str, x):
    if not isinstance(x, (int, float)):
        return x
    if key in NO_COMMA_KEYS:
        return f"{int(x)}" if float(x).is_integer() else f"{x}"
    if abs(x) >= 1000:
        if float(x).is_integer():
            return f"{int(x):,d}"
        return f"{x:,.2f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")

def pct_dev_from_ref(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    if value is None or ref is None:
        return None
    if not isinstance(value, (int, float)) or not isinstance(ref, (int, float)):
        return None
    if ref == 0:
        return None
    return (float(value) - float(ref)) / float(ref)

def threshold_pct_vs_ref(tol_abs: Optional[float], tol_rel: Optional[float], ref: Optional[float]) -> float:
    """Convert tolerance into a percent-of-reference threshold."""
    if not isinstance(ref, (int, float)) or ref == 0:
        return 0.05
    if isinstance(tol_rel, (int, float)):
        return float(tol_rel)
    if isinstance(tol_abs, (int, float)):
        return abs(float(tol_abs)) / abs(float(ref))
    return 0.05

def color_by_dev(dev: Optional[float], thresh: float) -> str:
    """Green within threshold, amber within 2×, red beyond."""
    if dev is None:
        return ""
    adev = abs(dev)
    if adev <= thresh:
        return "green"
    if adev <= 2 * thresh:
        return "#ff9800"  # amber
    return "red"

# ======================
# Rent table (Units # | Avg Rent | GPR)
# ======================
def build_rent_df(om: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k in RENT_KEYS:
        conf = RENT_METRICS[k]
        rent_key = conf["rent_key"]
        units_key = conf["units_key"]
        label = conf["label"]
        tol_abs = conf["tol_abs"]

        om_units = om.get(units_key)
        om_rent  = om.get(rent_key)
        r_rent   = realtor.get(rent_key)

        dev = pct_dev_from_ref(om_rent, r_rent)
        thresh = threshold_pct_vs_ref(tol_abs, None, r_rent)

        # GPRs (annual)
        gpr_om = None
        gpr_r  = None
        if isinstance(om_units, (int, float)):
            if isinstance(om_rent, (int, float)):
                gpr_om = om_units * om_rent * 12
            if isinstance(r_rent, (int, float)):
                gpr_r  = om_units * r_rent  * 12

        rows.append({
            "Unit Type": label,
            "Units #": int(om_units) if isinstance(om_units, (int, float)) else om_units,
            "Avg Rent": om_rent,
            "GPR": (gpr_om, gpr_r),
            "_dev": dev,
            "_thresh": thresh,
            "_rent_key": rent_key,
        })
    return pd.DataFrame(rows)

def style_rent_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format columns
    show["Units #"] = show["Units #"].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)
    show["Avg Rent"] = show["Avg Rent"].apply(fmt_money)
    def fmt_gpr(pair):
        if not isinstance(pair, tuple):
            return ""
        om, rr = pair
        om_s = fmt_money(om) if om is not None else "—"
        rr_s = fmt_money(rr) if rr is not None else "—"
        return f"OM: {om_s} | R: {rr_s}"
    show["GPR"] = show["GPR"].apply(fmt_gpr)

    styler = show[["Unit Type", "Units #", "Avg Rent", "GPR"]].style

    # Color bar on the LEFT of the Avg Rent cell (based on OM vs Realtor deviation)
    def left_bar_colors():
        styles = []
        for _, row in df.iterrows():
            col = color_by_dev(row["_dev"], row["_thresh"])
            if col:
                styles.append(f"border-left: 8px solid {col}; padding-left: 6px;")
            else:
                styles.append("")
        return styles

    # Apply styles
    styler = styler.apply(lambda _: left_bar_colors(), axis=0, subset=["Avg Rent"])

    # Zebra for readability
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# Location data table (only selected metrics compared)
# ======================
def build_location_df(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rule in LOC_METRICS:
        label = rule["label"]
        tol_abs = rule.get("tol_abs", None)
        tol_rel = rule.get("tol_rel", None)

        o = om.get(key)
        c = crexi.get(key)
        r = realtor.get(key)

        dev_r = pct_dev_from_ref(o, r)
        dev_c = pct_dev_from_ref(o, c)

        thresh_r = threshold_pct_vs_ref(tol_abs, tol_rel, r)
        thresh_c = threshold_pct_vs_ref(tol_abs, tol_rel, c)

        rows.append({
            "Metric": label,
            "OM": o,
            "Realtor": r,
            "OM vs Realtor Δ%": dev_r,
            "Crexi": c,
            "OM vs Crexi Δ%": dev_c,
            "_key": key,
            "_tol_abs": tol_abs,
            "_tol_rel": tol_rel,
            "_thresh_r": thresh_r,
            "_thresh_c": thresh_c,
        })
    return pd.DataFrame(rows)

def style_location_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format values appropriately
    for i, row in df.iterrows():
        key = row["_key"]
        for col in ["OM", "Realtor", "Crexi"]:
            show.at[i, col] = fmt_number(key, row[col])
        for col in ["OM vs Realtor Δ%", "OM vs Crexi Δ%"]:
            val = row[col]
            show.at[i, col] = f"{val*100:.1f}%" if isinstance(val, (int, float)) else ""

    styler = show[["Metric", "OM", "Realtor", "OM vs Realtor Δ%", "Crexi", "OM vs Crexi Δ%"]].style

    # Color the deviations
    def colors_vs_realtor():
        out = []
        for _, row in df.iterrows():
            out.append(f"background-color: {color_by_dev(row['OM vs Realtor Δ%'], row['_thresh_r'])};")
        return out

    def colors_vs_crexi():
        out = []
        for _, row in df.iterrows():
            out.append(f"background-color: {color_by_dev(row['OM vs Crexi Δ%'], row['_thresh_c'])};")
        return out

    styler = styler.apply(lambda _: colors_vs_realtor(), axis=0, subset=["OM vs Realtor Δ%"])
    styler = styler.apply(lambda _: colors_vs_crexi(), axis=0, subset=["OM vs Crexi Δ%"])

    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# UI
# ======================
st.title("🏗️ CRE Benchmarking Demo (Puppet)")
st.caption("Property summary + rent deviations (OM vs Realtor) and selected location comps (OM vs Realtor/Crexi). PDF drag-and-drop; tolerances collapsed.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded_pdf = st.file_uploader("Drag & drop the OM **PDF** (optional)", type=["pdf"])
    st.markdown("**Note:** Demo doesn't parse the PDF; it uses sample OM values to show the flow.")
    with st.expander("Adjust tolerances", expanded=False):
        # Allow adjusting rent tolerances
        for k in RENT_KEYS:
            conf = RENT_METRICS[k]
            new_abs = st.number_input(f"{conf['label']} rent tol_abs ($)", value=float(conf["tol_abs"]),
                                      step=5.0, key=f"tol_abs_rent_{k}")
            RENT_METRICS[k]["tol_abs"] = new_abs
        # Allow adjusting location tolerances
        for i, (key, rule) in enumerate(LOC_METRICS):
            if "tol_abs" in rule:
                new_val = st.number_input(f"{rule['label']} tol_abs",
                                          value=float(rule["tol_abs"]), step=1.0, key=f"tol_abs_loc_{key}")
                LOC_METRICS[i] = (key, {**rule, "tol_abs": new_val})
            elif "tol_rel" in rule:
                new_val = st.number_input(f"{rule['label']} tol_rel",
                                          value=float(rule["tol_rel"]), step=0.005, format="%.3f", key=f"tol_rel_loc_{key}")
                LOC_METRICS[i] = (key, {**rule, "tol_rel": new_val})

    run = st.button("Run Demo")

if run:
    om = parse_om(uploaded_pdf)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    # ---------- Property Summary ----------
    st.subheader("Property Summary")
    # Two-sentence blurb (example)
    blurb = (
        "123 Main St is a stabilized, garden-style multifamily asset located in a strong Tampa submarket "
        "with improving fundamentals and steady household growth. The property presents immediate upside "
        "via rent normalization to market and modest unit upgrades, with durable cash flow supported by "
        "solid occupancy."
    )
    # Render key fields in a neat grid
    cols = st.columns(3)
    summary_map = {
        "Address": om.get("address"),
        "OZ Status": om.get("oz_status"),
        "Avg Sq Ft / Unit": fmt_number("avg_sqft_per_type", om.get("avg_sqft_per_type")),
        "NOI": fmt_money(om.get("noi")),
        "Cap Rate": fmt_percent(om.get("cap_rate")),
        "Asking Price": fmt_money(om.get("asking_price")),
        "Expense Ratio": fmt_percent(om.get("expense_ratio")),
    }
    idx = 0
    for label, value in summary_map.items():
        with cols[idx % 3]:
            st.markdown(f"**{label}**")
            st.markdown(f"{value if value is not None else '—'}")
        idx += 1
    st.markdown(f"> {blurb}")

    # ---------- Rent Table ----------
    st.subheader("Unit Rents & GPR")
    rent_df_raw = build_rent_df(om, realtor)
    st.dataframe(style_rent_df(rent_df_raw), use_container_width=True, hide_index=True)

    # ---------- Location Data ----------
    st.subheader("Location Data (Selected Comparisons)")
    loc_df_raw = build_location_df(om, crexi, realtor)
    st.dataframe(style_location_df(loc_df_raw), use_container_width=True, hide_index=True)

else:
    st.info("Drag & drop an OM **PDF** (optional), set a query, adjust tolerances (expand the panel), then click **Run Demo**.")
