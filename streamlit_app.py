import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

st.set_page_config(page_title="CRE Scraping Demo (Puppet)", layout="wide")

# ======================
# Mock OM / Sources
# ======================
def parse_om(_uploaded_pdf) -> Dict[str, Any]:
    # Puppet OM values; rents intentionally higher than Realtor for demo
    return {
        "address": "123 Main St, Tampa, FL",
        # Unit Info
        "avg_rent_1bd": 1650, "units_1bd": 31,
        "avg_rent_2bd": 2100, "units_2bd": 52,
        "avg_rent_3bd": 2450, "units_3bd": 29,
        "avg_rent_4bd": 2800, "units_4bd": 8,
        "avg_sqft_per_type": 920,

        # Site / Summary
        "lot_size": 1.85,            # acres
        "year_built_or_renov": 2001, # summary-only
        "rentable_sqft": 108_750,
        "oz_status": "No",
        "total_units": 120,

        # Financials
        "noi": 1_890_000,
        "cap_rate": 0.054,
        "asking_price": 34_250_000,
    }

def fetch_crexi(query: str) -> Dict[str, Any]:
    # Fields aligned with what you can pull from Crexi
    return {
        "Property Link": "https://example.com/crexi/123",
        "Property Name": "Main Street Flats",
        "Property Status": "For Sale",
        "Type": "Multifamily",
        "Address": query or "123 Main St, Tampa, FL",
        "City": "Tampa",
        "State": "FL",
        "Zip": "33602",
        "SqFt": 107_200,            # rentable square feet
        "Lot Size": 1.70,           # acres
        "Units": 120,
        "Price/Unit": 275_000,      # $/unit
        "NOI": 1_825_000,
        "Cap Rate": 0.056,
        "Asking Price": 33_000_000,
        "Price/SqFt": 308,          # $/sqft
        "Price/Acre": 19_411_765,   # Asking / Lot Size
        "Opportunity Zone": "No",
        "Longitude": -82.4600,
        "Latitude": 27.9500,
        # extra for comparisons
        "avg_sqft_per_type": 910,
    }

def fetch_realtor(query: str) -> Dict[str, Any]:
    return {
        "address": query or "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1525, "avg_rent_2bd": 1935, "avg_rent_3bd": 2260, "avg_rent_4bd": 2550,
        "units_1bd": 29, "units_2bd": 49, "total_units": 118,
        "asking_price": 33_500_000,       # sometimes present; not required
        "avg_sqft_per_type": 900,
        "lot_size": 1.68,
        "rentable_sqft": 106_900,
        "oz_status": "No",
    }

# ======================
# Utilities / Formatting
# ======================
RENT_KEYS = ["1bd", "2bd", "3bd", "4bd"]
RENT_META = {
    "1bd": {"rent_key": "avg_rent_1bd", "units_key": "units_1bd", "label": "1 Bed"},
    "2bd": {"rent_key": "avg_rent_2bd", "units_key": "units_2bd", "label": "2 Bed"},
    "3bd": {"rent_key": "avg_rent_3bd", "units_key": "units_3bd", "label": "3 Bed"},
    "4bd": {"rent_key": "avg_rent_4bd", "units_key": "units_4bd", "label": "4 Bed"},
}

AREA_AVG_VACANCY = 0.06  # mock

def fmt_money(x):
    return f"${int(round(x)):,}" if isinstance(x, (int, float)) else x

def fmt_percent(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else x

def signed_pct(x: Optional[float]) -> str:
    if not isinstance(x, (int, float)):
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x)*100:.1f}%"

def pct_dev(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    if not (isinstance(value, (int, float)) and isinstance(ref, (int, float)) and ref != 0):
        return None
    return (float(value) - float(ref)) / float(ref)

def dev_color(dev: Optional[float]) -> str:
    if dev is None:
        return ""
    a = abs(dev)
    if a <= 0.05:      # ≤5% green
        return "#c8e6c9"
    if a <= 0.10:      # ≤10% amber
        return "#ffe0b2"
    return "#ffcdd2"   # >10% red

# ======================
# Comps synthesis + filtering (drives "Crexi Avg")
# ======================
def synthetic_comps(base: Dict[str, Any], n=300, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ask = float(base.get("Asking Price", 0) or 0)
    units = float(base.get("Units", 0) or 1)
    sqft = float(base.get("SqFt", 0) or 1)
    acres = float(base.get("Lot Size", 0) or 1)
    cap = float(base.get("Cap Rate", 0.05) or 0.05)
    noi = float(base.get("NOI", ask * cap) or ask * cap)

    # jitter scales
    ask_sigma = 0.12   # 12%
    units_sigma = 0.20 # ±20% of units in count space
    sqft_sigma = 0.12
    acres_sigma = 0.18
    cap_sigma = 0.008  # ±80 bps
    noi_sigma = 0.10
    dist_max = 25.0

    df = pd.DataFrame({
        "asking_price": ask * (1 + rng.normal(0, ask_sigma, n)),
        "units": np.maximum(1, np.round(units * (1 + rng.normal(0, units_sigma, n)))).astype(int),
        "sqft": np.maximum(5000, sqft * (1 + rng.normal(0, sqft_sigma, n))),
        "acres": np.maximum(0.1, acres * (1 + rng.normal(0, acres_sigma, n))),
        "cap_rate": np.clip(cap + rng.normal(0, cap_sigma, n), 0.02, 0.15),
        "noi": np.maximum(0, noi * (1 + rng.normal(0, noi_sigma, n))),
        "distance_miles": rng.uniform(0, dist_max, n)
    })
    df["price_per_sqft"] = df["asking_price"] / df["sqft"]
    df["price_per_unit"] = df["asking_price"] / df["units"]
    df["price_per_acre"] = df["asking_price"] / df["acres"]
    return df

def apply_filters(df: pd.DataFrame, subject: Dict[str, Any],
                  max_price_pct: float,
                  max_units_diff: float,
                  max_distance: float,
                  max_cap_pp: float,
                  max_ppu_pct: float,
                  max_ppsf_pct: float,
                  max_ppa_pct: float) -> pd.DataFrame:
    # Subject reference values
    s_ask = subject.get("asking_price")
    s_units = subject.get("total_units")
    s_sqft = subject.get("rentable_sqft")
    s_acres = subject.get("lot_size")
    s_cap = subject.get("cap_rate")

    # Derived subject $/metrics
    s_ppsf = s_ask / s_sqft if isinstance(s_ask,(int,float)) and isinstance(s_sqft,(int,float)) and s_sqft else None
    s_ppu  = s_ask / s_units if isinstance(s_ask,(int,float)) and isinstance(s_units,(int,float)) and s_units else None
    s_ppa  = s_ask / s_acres if isinstance(s_ask,(int,float)) and isinstance(s_acres,(int,float)) and s_acres else None

    m = pd.Series(True, index=df.index)

    if isinstance(s_ask,(int,float)):
        m &= (abs(df["asking_price"] - s_ask) / s_ask) <= max_price_pct
    if isinstance(s_units,(int,float)):
        m &= abs(df["units"] - s_units) <= max_units_diff
    m &= df["distance_miles"] <= max_distance
    if isinstance(s_cap,(int,float)):
        m &= abs(df["cap_rate"] - s_cap) <= max_cap_pp

    if isinstance(s_ppu,(int,float)):
        m &= (abs(df["price_per_unit"] - s_ppu) / s_ppu) <= max_ppu_pct
    if isinstance(s_ppsf,(int,float)):
        m &= (abs(df["price_per_sqft"] - s_ppsf) / s_ppsf) <= max_ppsf_pct
    if isinstance(s_ppa,(int,float)):
        m &= (abs(df["price_per_acre"] - s_ppa) / s_ppa) <= max_ppa_pct

    return df[m]

def crexi_avgs_from_filtered(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if df.empty:
        return {k: None for k in [
            "Asking Price","NOI","Cap Rate","Price/SqFt","Price/Unit","Price/Acre",
            "SqFt","Units","Lot Size"
        ]}
    return {
        "Asking Price": float(df["asking_price"].mean()),
        "NOI": float(df["noi"].mean()),
        "Cap Rate": float(df["cap_rate"].mean()),
        "Price/SqFt": float(df["price_per_sqft"].mean()),
        "Price/Unit": float(df["price_per_unit"].mean()),
        "Price/Acre": float(df["price_per_acre"].mean()),
        "SqFt": float(df["sqft"].mean()),
        "Units": float(df["units"].mean()),
        "Lot Size": float(df["acres"].mean()),
    }

# ======================
# Rent table + totals (weighted by unit count)
# ======================
def build_rent_df(om: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k in RENT_KEYS:
        meta = RENT_META[k]
        rent_key, units_key, label = meta["rent_key"], meta["units_key"], meta["label"]
        om_units = om.get(units_key)
        om_rent  = om.get(rent_key)
        r_rent   = realtor.get(rent_key)

        # average baseline for % deviation = mean(OM, Realtor) when both present
        vals = [v for v in [om_rent, r_rent] if isinstance(v, (int, float))]
        avg_rent = float(np.mean(vals)) if vals else None
        dev_vs_avg = pct_dev(om_rent, avg_rent)

        # GPRs (annual) using OM units
        gpr_om = gpr_r = None
        if isinstance(om_units, (int, float)):
            if isinstance(om_rent, (int, float)):
                gpr_om = om_units * om_rent * 12
            if isinstance(r_rent, (int, float)):
                gpr_r  = om_units * r_rent  * 12

        rows.append({
            "Unit Type": label,
            "Units #": int(om_units) if isinstance(om_units, (int, float)) else om_units,
            "Avg Rent": (r_rent, dev_vs_avg),
            "GPR": (gpr_om, gpr_r),
            "_units": om_units, "_om_rent": om_rent, "_r_rent": r_rent
        })
    return pd.DataFrame(rows)

def compute_property_wide(rent_df_raw: pd.DataFrame):
    # Weighted by OM units
    total_units = 0
    wsum_om = 0.0
    wsum_r  = 0.0
    gpr_om = 0.0
    gpr_r  = 0.0
    for _, row in rent_df_raw.iterrows():
        u = row["_units"]; om_r = row["_om_rent"]; r_r = row["_r_rent"]
        if isinstance(u, (int,float)):
            total_units += int(u)
            if isinstance(om_r, (int,float)):
                wsum_om += om_r * u
                gpr_om  += u * om_r * 12
            if isinstance(r_r, (int,float)):
                wsum_r  += r_r * u
                gpr_r   += u * r_r * 12
    avg_om = (wsum_om / total_units) if total_units and wsum_om else None
    avg_r  = (wsum_r  / total_units) if total_units and wsum_r  else None
    # deviation relative to average of the two weighted averages
    vals = [v for v in [avg_om, avg_r] if isinstance(v, (int,float))]
    avg_pair = float(np.mean(vals)) if vals else None
    dev_vs_avg_total = pct_dev(avg_om, avg_pair)
    return {
        "total_units": total_units,
        "avg_om": avg_om,
        "avg_r": avg_r,
        "gpr_om": gpr_om if gpr_om else None,
        "gpr_r": gpr_r if gpr_r else None,
        "dev_vs_avg_total": dev_vs_avg_total
    }

def style_rent_df(df: pd.DataFrame, totals: Dict[str, Any]) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format visible fields
    show["Units #"] = show["Units #"].apply(lambda x: f"{int(x)}" if isinstance(x,(int,float)) else x)

    def fmt_avg(cell):
        if not isinstance(cell, tuple): return ""
        r_rent, dev = cell
        r_str = fmt_money(r_rent) if isinstance(r_rent,(int,float)) else "—"
        return f"R: {r_str} | OM vs avg: {signed_pct(dev)}"

    def fmt_gpr(cell):
        if not isinstance(cell, tuple): return ""
        om, rr = cell
        om_s = fmt_money(om) if isinstance(om,(int,float)) else "—"
        rr_s = fmt_money(rr) if isinstance(rr,(int,float)) else "—"
        return f"OM: {om_s} | R: {rr_s}"

    show["Avg Rent"] = show["Avg Rent"].apply(fmt_avg)
    show["GPR"] = show["GPR"].apply(fmt_gpr)

    # Totals row (weighted by OM units)
    totals_row = {
        "Unit Type": "All Units (property-wide)",
        "Units #": f"{totals['total_units']:,}" if totals["total_units"] else "—",
        "Avg Rent": f"OM: {fmt_money(totals['avg_om']) if totals['avg_om'] else '—'} | "
                    f"R: {fmt_money(totals['avg_r']) if totals['avg_r'] else '—'} | "
                    f"OM vs avg: {signed_pct(totals['dev_vs_avg_total'])}",
        "GPR": f"OM: {fmt_money(totals['gpr_om']) if totals['gpr_om'] else '—'} | "
               f"R: {fmt_money(totals['gpr_r']) if totals['gpr_r'] else '—'}"
    }
    show = pd.concat([show[["Unit Type","Units #","Avg Rent","GPR"]],
                      pd.DataFrame([totals_row])], ignore_index=True)

    styler = show.style

    # Colorize Avg Rent column (per row; totals uses a neutral color band)
    def avg_rent_colors():
        styles = []
        n = len(df)
        for i in range(len(show)):
            if i < n:
                r_rent, dev = df.iloc[i]["Avg Rent"]
                styles.append(f"background-color: {dev_color(dev)};")
            else:
                styles.append("")  # totals: no fill, readable
        return styles

    styler = styler.apply(lambda _: avg_rent_colors(), axis=0, subset=["Avg Rent"])
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# Financials table (uses filtered Crexi averages)
# ======================
def compute_price_metrics(asking: Optional[float], sqft: Optional[float],
                          units: Optional[float], acres: Optional[float]):
    pps = asking / sqft if isinstance(asking,(int,float)) and isinstance(sqft,(int,float)) and sqft else None
    ppu = asking / units if isinstance(asking,(int,float)) and isinstance(units,(int,float)) and units else None
    ppa = asking / acres if isinstance(asking,(int,float)) and isinstance(acres,(int,float)) and acres else None
    return pps, ppu, ppa

def build_financials_table(om: Dict[str, Any], crexi_avg: Dict[str, Any]) -> pd.DataFrame:
    om_price = om.get("asking_price")
    om_noi   = om.get("noi")
    om_cap   = om.get("cap_rate")
    om_sqft  = om.get("rentable_sqft")
    om_units = om.get("total_units")
    om_acres = om.get("lot_size")

    om_pps, om_ppu, om_ppa = compute_price_metrics(om_price, om_sqft, om_units, om_acres)

    cx_price = crexi_avg.get("Asking Price")
    cx_noi   = crexi_avg.get("NOI")
    cx_cap   = crexi_avg.get("Cap Rate")
    cx_pps   = crexi_avg.get("Price/SqFt")
    cx_ppu   = crexi_avg.get("Price/Unit")
    cx_ppa   = crexi_avg.get("Price/Acre")

    rows = [
        ("Asking Price", om_price, cx_price, pct_dev(om_price, cx_price), fmt_money),
        ("Cap Rate",     om_cap,   cx_cap,   pct_dev(om_cap,   cx_cap),   fmt_percent),
        ("NOI",          om_noi,   cx_noi,   pct_dev(om_noi,   cx_noi),   fmt_money),
        ("Price/SqFt",   om_pps,   cx_pps,   pct_dev(om_pps,   cx_pps),   fmt_money),
        ("Price/Unit",   om_ppu,   cx_ppu,   pct_dev(om_ppu,   cx_ppu),   fmt_money),
        ("Price/Acre",   om_ppa,   cx_ppa,   pct_dev(om_ppa,   cx_ppa),   fmt_money),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "OM", "Crexi Avg", "Deviation", "_fmt"])
    # Style-ready
    show = df.copy()
    for i, row in df.iterrows():
        f = row["_fmt"]
        show.at[i,"OM"] = f(row["OM"]) if callable(f) and isinstance(row["OM"],(int,float)) else row["OM"]
        show.at[i,"Crexi Avg"] = f(row["Crexi Avg"]) if callable(f) and isinstance(row["Crexi Avg"],(int,float)) else row["Crexi Avg"]
        show.at[i,"Deviation"] = signed_pct(row["Deviation"])
    styler = show[["Metric","OM","Crexi Avg","Deviation"]].style
    def colors():
        out=[]
        for _, r in df.iterrows():
            out.append(f"background-color: {dev_color(r['Deviation'])};" if isinstance(r["Deviation"],(int,float)) else "")
        return out
    return styler.apply(lambda _: colors(), axis=0, subset=["Deviation"])

# ======================
# All metrics table (OM, Crexi Avg (filtered), Realtor)
# For Address, Units, OZ, Lot Size -> OM only (blank others)
# ======================
def build_all_metrics_table(om: Dict[str, Any], crexi_avg: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    def add(name, om_val=None, cx_val=None, r_val=None, fmt=None,
            om_only=False):
        cx = None if om_only else cx_val
        r  = None if om_only else r_val
        if fmt and isinstance(om_val,(int,float)): om_val = fmt(om_val)
        if fmt and isinstance(cx,(int,float)):      cx     = fmt(cx)
        if fmt and isinstance(r,(int,float)):       r      = fmt(r)
        rows.append([name, om_val, cx, r])

    add("Address", om.get("address"), om_only=True)
    add("OZ Status", om.get("oz_status"), om_only=True)
    add("Total Units", om.get("total_units"), om_only=True, fmt=lambda v: int(round(v)))
    add("Lot Size (acres)", om.get("lot_size"), om_only=True, fmt=lambda v: float(v))

    add("Rentable Sq Ft", om.get("rentable_sqft"), crexi_avg.get("SqFt"), realtor.get("rentable_sqft"), fmt=lambda v: int(round(v)))
    add("Avg Sq Ft / Unit", om.get("avg_sqft_per_type"), crexi_avg.get("SqFt")/om.get("total_units") if isinstance(crexi_avg.get("SqFt"),(int,float)) and isinstance(om.get("total_units"),(int,float)) and om.get("total_units")!=0 else None, realtor.get("avg_sqft_per_type"), fmt=lambda v: int(round(v)) if isinstance(v,(int,float)) else v)

    add("Asking Price", om.get("asking_price"), crexi_avg.get("Asking Price"), realtor.get("asking_price"), fmt=fmt_money)
    add("NOI", om.get("noi"), crexi_avg.get("NOI"), None, fmt=fmt_money)
    add("Cap Rate", om.get("cap_rate"), crexi_avg.get("Cap Rate"), None, fmt=fmt_percent)

    # Rents (OM/Realtor only)
    add("Avg Rent (1 Bed)", om.get("avg_rent_1bd"), None, realtor.get("avg_rent_1bd"), fmt=fmt_money)
    add("Avg Rent (2 Bed)", om.get("avg_rent_2bd"), None, realtor.get("avg_rent_2bd"), fmt=fmt_money)
    add("Avg Rent (3 Bed)", om.get("avg_rent_3bd"), None, realtor.get("avg_rent_3bd"), fmt=fmt_money)
    add("Avg Rent (4 Bed)", om.get("avg_rent_4bd"), None, realtor.get("avg_rent_4bd"), fmt=fmt_money)

    # Derived prices
    def drv(ask, sqft, units, acres):
        pps = ask/sqft if isinstance(ask,(int,float)) and isinstance(sqft,(int,float)) and sqft else None
        ppu = ask/units if isinstance(ask,(int,float)) and isinstance(units,(int,float)) and units else None
        ppa = ask/acres if isinstance(ask,(int,float)) and isinstance(acres,(int,float)) and acres else None
        return pps, ppu, ppa
    om_pps, om_ppu, om_ppa = drv(om.get("asking_price"), om.get("rentable_sqft"), om.get("total_units"), om.get("lot_size"))
    cx_pps, cx_ppu, cx_ppa = crexi_avg.get("Price/SqFt"), crexi_avg.get("Price/Unit"), crexi_avg.get("Price/Acre")

    rows.append(["Price/SqFt", fmt_money(om_pps) if om_pps else None, fmt_money(cx_pps) if cx_pps else None, None])
    rows.append(["Price/Unit", fmt_money(om_ppu) if om_ppu else None, fmt_money(cx_ppu) if cx_ppu else None, None])
    rows.append(["Price/Acre", fmt_money(om_ppa) if om_ppa else None, fmt_money(cx_ppa) if cx_ppa else None, None])

    return pd.DataFrame(rows, columns=["Metric","OM","Crexi (Filtered Avg)","Realtor"])

# ======================
# UI
# ======================
st.title("🏗️ CRE Scraping Demo (Puppet)")
st.caption("Property summary, unit rents (weighted averages), Financials (using filtered Crexi comps), Vacancy & EGI, and an all-metrics rollup. PDF upload is mocked.")

with st.sidebar:
    st.header("Inputs")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded_pdf = st.file_uploader("Drag & drop the OM **PDF** (optional)", type=["pdf"])

    st.header("Comps Filters")
    col_a, col_b = st.columns(2)
    with col_a:
        max_price_pct = st.number_input("Max % diff: Asking Price", value=0.10, step=0.01, format="%.2f")
        max_units_diff = st.number_input("Max Units difference (count)", value=20.0, step=1.0)
        max_cap_pp = st.number_input("Max Cap Rate difference (pp)", value=0.010, step=0.002, format="%.3f")
    with col_b:
        max_distance = st.number_input("Max Distance (miles)", value=15.0, step=1.0)
        max_ppu_pct = st.number_input("Max % diff: Price/Unit", value=0.10, step=0.01, format="%.2f")
        max_ppsf_pct = st.number_input("Max % diff: Price/SqFt", value=0.10, step=0.01, format="%.2f")
        max_ppa_pct = st.number_input("Max % diff: Price/Acre", value=0.15, step=0.01, format="%.2f")

    with st.expander("Advanced (synthetic comps)", expanded=False):
        n_comps = st.slider("Number of synthetic comps", 50, 1000, 300, step=50)
        seed = st.number_input("Random seed", value=7, step=1)

    run = st.button("Run Demo")

if run:
    # Data
    om = parse_om(uploaded_pdf)
    cx = fetch_crexi(query)
    r  = fetch_realtor(query)

    # ---------- Property Summary ----------
    st.subheader("Property Summary")
    blurb = (
        "123 Main St is a stabilized, garden-style multifamily asset in a strong Tampa submarket. "
        "There is opportunity to align OM rents with market while maintaining durable occupancy."
    )
    cols = st.columns(3)
    summary = {
        "Address": om.get("address"),
        "OZ Status": om.get("oz_status"),
        "Cap Rate": fmt_percent(om.get("cap_rate")),
        "Rentable Sq Ft": f"{int(round(om.get('rentable_sqft'))):,}" if isinstance(om.get("rentable_sqft"),(int,float)) else "—",
        "Avg Sq Ft / Unit": f"{int(round(om.get('avg_sqft_per_type'))):,}" if isinstance(om.get("avg_sqft_per_type"),(int,float)) else "—",
        "Asking Price": fmt_money(om.get("asking_price")),
        "Year Built / Renovated": f"{int(round(om.get('year_built_or_renov')))}" if isinstance(om.get("year_built_or_renov"),(int,float)) else "—",
        "Lot Size (acres)": f"{om.get('lot_size'):.2f}" if isinstance(om.get("lot_size"),(int,float)) else "—",
        "Total Units": f"{int(round(om.get('total_units')))}" if isinstance(om.get("total_units"),(int,float)) else "—",
    }
    i = 0
    for k, v in summary.items():
        with cols[i % 3]:
            st.markdown(f"**{k}**")
            st.markdown(v if v is not None else "—")
        i += 1
    st.markdown(f"> {blurb}")

    # ---------- Build & filter synthetic comps -> filtered Crexi averages ----------
    comps = synthetic_comps(cx, n=n_comps, seed=int(seed))
    filtered = apply_filters(
        comps, om,
        max_price_pct=max_price_pct,
        max_units_diff=max_units_diff,
        max_distance=max_distance,
        max_cap_pp=max_cap_pp,
        max_ppu_pct=max_ppu_pct,
        max_ppsf_pct=max_ppsf_pct,
        max_ppa_pct=max_ppa_pct,
    )
    crexi_avg = crexi_avgs_from_filtered(filtered)

    with st.expander("Filtered comps summary", expanded=False):
    st.write(f"Comps after filters: **{len(filtered)} / {len(comps)}**")
    if filtered.empty:
        st.info("No comps passed your current filters. Loosen one or more sliders to see stats.")
    else:
        # Select numeric columns explicitly to avoid pandas version issues with numeric_only
        num_only = filtered.select_dtypes(include=[np.number])
        # If for some reason there are no numeric columns left, show the head instead
        if num_only.shape[1] == 0:
            st.dataframe(filtered.head(20), use_container_width=True, hide_index=True)
        else:
            st.dataframe(num_only.describe().T, use_container_width=True)

    # ---------- Rent table (with weighted totals) ----------
    st.subheader("Unit Rents & GPR (OM vs Realtor)")
    rent_df_raw = build_rent_df(om, r)
    totals = compute_property_wide(rent_df_raw)
    st.dataframe(style_rent_df(rent_df_raw, totals), use_container_width=True, hide_index=True)

    # ---------- Financials (uses filtered Crexi averages) ----------
    st.subheader("Financials")
    fin_styler = build_financials_table(om, crexi_avg)
    st.dataframe(fin_styler, use_container_width=True, hide_index=True)

    # ---------- Vacancy & EGI ----------
    st.subheader("Area Average Vacancy")
    st.markdown(f"**{fmt_percent(AREA_AVG_VACANCY)}**")
    # EGI = GPR × (1 − vacancy) using property-wide GPR from OM and Realtor rents
    gpr_om = totals["gpr_om"]; gpr_r = totals["gpr_r"]
    egi_om = gpr_om*(1-AREA_AVG_VACANCY) if isinstance(gpr_om,(int,float)) else None
    egi_r  = gpr_r *(1-AREA_AVG_VACANCY) if isinstance(gpr_r,(int,float))  else None
    st.markdown("EGI (Effective Gross Income) is computed as **GPR × (1 − vacancy)** using the area average vacancy rate.")
    egi_tbl = pd.DataFrame([
        ["GPR (OM rents)", fmt_money(gpr_om) if gpr_om else "—"],
        ["GPR (Realtor rents)", fmt_money(gpr_r) if gpr_r else "—"],
        ["EGI (OM rents)", fmt_money(egi_om) if egi_om else "—"],
        ["EGI (Realtor rents)", fmt_money(egi_r) if egi_r else "—"],
    ], columns=["Metric","Value"])
    st.dataframe(egi_tbl, use_container_width=True, hide_index=True)

    # ---------- All Metrics (OM, Crexi filtered avg, Realtor) ----------
    st.subheader("All Metrics (OM, Crexi Filtered Avg, Realtor)")
    all_df = build_all_metrics_table(om, crexi_avg, r)
    st.dataframe(all_df, use_container_width=True, hide_index=True)

else:
    st.info("Set filters, drop an OM **PDF** (optional), then click **Run Demo**.")
