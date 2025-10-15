import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
    Puppet OM parser (PDF upload ignored; replace with real parser later).
    OM rents intentionally high vs Realtor for demo.
    """
    return {
        "address": "123 Main St, Tampa, FL",
        # Unit Info (rents high vs Realtor)
        "avg_rent_1bd": 1650, "units_1bd": 31,
        "avg_rent_2bd": 2100, "units_2bd": 52,
        "avg_rent_3bd": 2450, "units_3bd": 29,
        "avg_rent_4bd": 2800, "units_4bd": 8,
        "avg_sqft_per_type": 920,

        # Summary / site
        "lot_size": 1.85,            # acres
        "year_built_or_renov": 2001,
        "rentable_sqft": 108_750,
        "oz_status": "No",
        "total_units": 120,

        # Financials
        "noi": 1_890_000,
        "cap_rate": 0.054,
        "asking_price": 34_250_000,
        # no expense ratio
    }

def fetch_crexi(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fields aligned to what you can pull from Crexi (per your list).
    """
    if mock is not None:
        return mock
    return {
        "Property Link": "https://example.com/crexi/123",   # placeholder
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

def fetch_realtor(query: str, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if mock is not None:
        return mock
    return {
        "address": query or "123 Main St, Tampa, FL",
        "avg_rent_1bd": 1525, "avg_rent_2bd": 1935, "avg_rent_3bd": 2260, "avg_rent_4bd": 2550,
        "units_1bd": 29, "units_2bd": 49, "total_units": 118,
        "asking_price": 33_500_000,    # may exist or be missing
        "avg_sqft_per_type": 900,
        "lot_size": 1.68,
        "rentable_sqft": 106_900,
        "oz_status": "No",
    }

# ======================
# Config / tolerances
# ======================
RENT_KEYS = ["1bd", "2bd", "3bd", "4bd"]
RENT_METRICS = {
    "1bd": {"rent_key": "avg_rent_1bd", "units_key": "units_1bd", "label": "1 Bed", "tol_abs": 50},
    "2bd": {"rent_key": "avg_rent_2bd", "units_key": "units_2bd", "label": "2 Bed", "tol_abs": 75},
    "3bd": {"rent_key": "avg_rent_3bd", "units_key": "units_3bd", "label": "3 Bed", "tol_abs": 100},
    "4bd": {"rent_key": "avg_rent_4bd", "units_key": "units_4bd", "label": "4 Bed", "tol_abs": 120},
}

NO_COMMA_KEYS = {"year_built_or_renov"}

# Vacancy area average (mock)
AREA_AVG_VACANCY = 0.06

# ======================
# Formatting + helpers
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
        return f"{int(round(x))}" if float(x).is_integer() else f"{x}"
    if abs(x) >= 1000:
        if float(x).is_integer():
            return f"{int(round(x)):,d}"
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
    """Return CSS color string (green / amber / red)."""
    if dev is None:
        return ""
    adev = abs(dev)
    if adev <= thresh:
        return "green"
    if adev <= 2 * thresh:
        return "#ff9800"
    return "red"

def signed_pct(x: Optional[float]) -> str:
    if not isinstance(x, (int, float)):
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x)*100:.1f}%"

# ======================
# Rent math & table
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

        # average = mean(OM rent, Realtor rent) if both present, else the one that exists
        vals = [v for v in [om_rent, r_rent] if isinstance(v, (int, float))]
        avg_rent = float(np.mean(vals)) if vals else None
        dev_vs_avg = pct_dev_from_ref(om_rent, avg_rent)

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
            "Avg Rent": (r_rent, dev_vs_avg, avg_rent, tol_abs),
            "GPR": (gpr_om, gpr_r),
            "_units": om_units, "_om_rent": om_rent, "_r_rent": r_rent
        })
    return pd.DataFrame(rows)

def compute_property_wide(rent_df_raw: pd.DataFrame):
    # Weighted by OM units; totals GPR use OM units
    total_units = 0
    wsum_om = 0.0
    wsum_r  = 0.0
    total_gpr_om = 0.0
    total_gpr_r  = 0.0
    for _, row in rent_df_raw.iterrows():
        u = row["_units"]
        om_r = row["_om_rent"]
        r_r  = row["_r_rent"]
        if isinstance(u, (int,float)):
            total_units += int(u)
            if isinstance(om_r, (int,float)):
                wsum_om += om_r * u
                total_gpr_om += u * om_r * 12
            if isinstance(r_r, (int,float)):
                wsum_r  += r_r * u
                total_gpr_r  += u * r_r * 12
    avg_om = (wsum_om / total_units) if total_units and wsum_om else None
    avg_r  = (wsum_r  / total_units) if total_units and wsum_r  else None
    # % deviation vs avg of the two weighted avgs (if both exist; else vs whichever exists)
    vals = [v for v in [avg_om, avg_r] if isinstance(v, (int,float))]
    avg_pair = float(np.mean(vals)) if vals else None
    dev_vs_avg_total = pct_dev_from_ref(avg_om, avg_pair)
    return {
        "total_units": total_units,
        "avg_om": avg_om,
        "avg_r": avg_r,
        "gpr_om": total_gpr_om if total_gpr_om else None,
        "gpr_r": total_gpr_r if total_gpr_r else None,
        "dev_vs_avg_total": dev_vs_avg_total
    }

def style_rent_df(df: pd.DataFrame, totals: Dict[str, Any]) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format columns (per-row)
    show["Units #"] = show["Units #"].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)

    def fmt_avg_rent(cell):
        if not isinstance(cell, tuple) or len(cell) < 4:
            return ""
        r_rent, dev_vs_avg, avg_rent, tol_abs = cell
        r_str = fmt_money(r_rent) if isinstance(r_rent, (int, float)) else "—"
        d_str = signed_pct(dev_vs_avg)
        return f"R: {r_str} | OM vs avg: {d_str}"

    show["Avg Rent"] = show["Avg Rent"].apply(fmt_avg_rent)

    def fmt_gpr(pair):
        if not isinstance(pair, tuple):
            return ""
        om, rr = pair
        om_s = fmt_money(om) if isinstance(om, (int, float)) else "—"
        rr_s = fmt_money(rr) if isinstance(rr, (int, float)) else "—"
        return f"OM: {om_s} | R: {rr_s}"

    show["GPR"] = show["GPR"].apply(fmt_gpr)

    # Append totals row (property-wide)
    totals_row = {
        "Unit Type": "All Units (property-wide)",
        "Units #": f"{totals['total_units']:,}" if totals["total_units"] else "—",
        "Avg Rent": f"OM: {fmt_money(totals['avg_om']) if totals['avg_om'] else '—'} | "
                    f"R: {fmt_money(totals['avg_r']) if totals['avg_r'] else '—'} | "
                    f"OM vs avg: {signed_pct(totals['dev_vs_avg_total'])}",
        "GPR": f"OM: {fmt_money(totals['gpr_om']) if totals['gpr_om'] else '—'} | "
               f"R: {fmt_money(totals['gpr_r'])  if totals['gpr_r']  else '—'}"
    }
    show = pd.concat([show[["Unit Type","Units #","Avg Rent","GPR"]], pd.DataFrame([totals_row])], ignore_index=True)

    styler = show.style

    # Left color bar on per-row Avg Rent cells based on OM vs avg Δ%
    def left_bar_styles():
        styles = []
        n = len(df)
        for i in range(len(show)):
            if i < n:
                r_rent, dev_vs_avg, avg_rent, tol_abs = df.iloc[i]["Avg Rent"]
                thresh = threshold_pct_vs_ref(tol_abs, None, avg_rent)
                col = color_by_dev(dev_vs_avg, thresh)
                styles.append(f"border-left: 8px solid {col}; padding-left: 6px;" if col else "")
            else:
                # totals row: color based on totals deviation using tolerance ~
                # take a rough tol_abs proxy: mean of per-type tolerances
                tol_abs_mean = np.mean([RENT_METRICS[k]["tol_abs"] for k in RENT_KEYS])
                thresh = threshold_pct_vs_ref(tol_abs_mean, None, np.mean([v for v in [totals['avg_om'], totals['avg_r']] if isinstance(v,(int,float))]) if any(isinstance(v,(int,float)) for v in [totals['avg_om'], totals['avg_r']]) else None)
                col = color_by_dev(totals['dev_vs_avg_total'], thresh)
                styles.append(f"border-left: 8px solid {col}; padding-left: 6px;" if col else "")
        return styles

    styler = styler.apply(lambda _: left_bar_styles(), axis=0, subset=["Avg Rent"])

    # Zebra
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# Financials table (no vacancy row here)
# ======================
def compute_price_metrics(asking: Optional[float], sqft: Optional[float], units: Optional[float], acres: Optional[float]):
    price_per_sqft = asking / sqft if isinstance(asking, (int,float)) and isinstance(sqft, (int,float)) and sqft != 0 else None
    price_per_unit = asking / units if isinstance(asking, (int,float)) and isinstance(units, (int,float)) and units != 0 else None
    price_per_acre = asking / acres if isinstance(asking, (int,float)) and isinstance(acres, (int,float)) and acres != 0 else None
    return price_per_sqft, price_per_unit, price_per_acre

def build_financials_table(om: Dict[str, Any], crexi: Dict[str, Any]) -> pd.DataFrame:
    om_price = om.get("asking_price")
    cx_price = crexi.get("Asking Price")
    om_noi   = om.get("noi")
    cx_noi   = crexi.get("NOI")
    om_cap   = om.get("cap_rate")
    cx_cap   = crexi.get("Cap Rate")

    om_sqft  = om.get("rentable_sqft")
    cx_sqft  = crexi.get("SqFt")
    om_units = om.get("total_units")
    cx_units = crexi.get("Units")
    om_acres = om.get("lot_size")
    cx_acres = crexi.get("Lot Size")

    om_pps, om_ppu, om_ppa = compute_price_metrics(om_price, om_sqft, om_units, om_acres)
    cx_pps, cx_ppu, cx_ppa = compute_price_metrics(cx_price, cx_sqft, cx_units, cx_acres)

    def dev(a, b):  # signed percent (OM vs Crexi)
        return pct_dev_from_ref(a, b)

    rows = [
        ("Asking Price", om_price, cx_price, dev(om_price, cx_price), fmt_money),
        ("Cap Rate",     om_cap,   cx_cap,   dev(om_cap,   cx_cap),   fmt_percent),
        ("NOI",          om_noi,   cx_noi,   dev(om_noi,   cx_noi),   fmt_money),
        ("Price/SqFt",   om_pps,   cx_pps,   dev(om_pps,   cx_pps),   lambda v: fmt_money(v) if isinstance(v,(int,float)) else v),
        ("Price/Unit",   om_ppu,   cx_ppu,   dev(om_ppu,   cx_ppu),   fmt_money),
        ("Price/Acre",   om_ppa,   cx_ppa,   dev(om_ppa,   cx_ppa),   fmt_money),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "OM", "Crexi Avg", "Deviation", "_fmt"])
    return df

def style_financials_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()
    for i, row in df.iterrows():
        f = row["_fmt"]
        show.at[i, "OM"] = f(row["OM"]) if callable(f) else row["OM"]
        show.at[i, "Crexi Avg"] = f(row["Crexi Avg"]) if callable(f) else row["Crexi Avg"]
        show.at[i, "Deviation"] = signed_pct(row["Deviation"])
    styler = show[["Metric", "OM", "Crexi Avg", "Deviation"]].style
    def colors_for_delta():
        out = []
        for _, row in df.iterrows():
            d = row["Deviation"]
            out.append(f"background-color: {color_by_dev(d, 0.05)};" if isinstance(d,(int,float)) else "")
        return out
    styler = styler.apply(lambda _: colors_for_delta(), axis=0, subset=["Deviation"])
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# “All Metrics” combined table (OM, Crexi, Realtor)
# with blanks for certain rows on Crexi/Realtor as requested
# ======================
def build_all_metrics_table(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    def add_row(name, om_key=None, cx_key=None, r_key=None, fmt=None, force_blank_cr=False):
        om_val = om.get(om_key) if om_key else None
        cx_val = crexi.get(cx_key) if (cx_key and not force_blank_cr) else None
        r_val  = realtor.get(r_key) if (r_key and not force_blank_cr) else None

        if fmt and isinstance(om_val,(int,float)): om_val = fmt(om_val)
        if fmt and isinstance(cx_val,(int,float)): cx_val = fmt(cx_val)
        if fmt and isinstance(r_val,(int,float)):  r_val  = fmt(r_val)

        rows.append([name, om_val, cx_val, r_val])

    # Rows where ONLY OM should appear (address, units, OZ, lot size)
    add_row("Address", "address", None, None, None, force_blank_cr=True)
    add_row("OZ Status", "oz_status", None, None, None, force_blank_cr=True)
    add_row("Total Units", "total_units", None, None, lambda v: f"{int(round(v))}", force_blank_cr=True)
    add_row("Lot Size (acres)", "lot_size", None, None, lambda v: f"{v:.2f}", force_blank_cr=True)

    # Other summary / financials / rents
    add_row("Rentable Sq Ft", "rentable_sqft", "SqFt", "rentable_sqft", lambda v: f"{int(round(v)):,}")
    add_row("Avg Sq Ft / Unit", "avg_sqft_per_type", "avg_sqft_per_type", "avg_sqft_per_type", lambda v: f"{int(round(v)):,}")
    add_row("Asking Price", "asking_price", "Asking Price", "asking_price", fmt_money)
    add_row("NOI", "noi", "NOI", None, fmt_money)
    add_row("Cap Rate", "cap_rate", "Cap Rate", None, fmt_percent)

    add_row("Avg Rent (1 Bed)", "avg_rent_1bd", None, "avg_rent_1bd", fmt_money)
    add_row("Avg Rent (2 Bed)", "avg_rent_2bd", None, "avg_rent_2bd", fmt_money)
    add_row("Avg Rent (3 Bed)", "avg_rent_3bd", None, "avg_rent_3bd", fmt_money)
    add_row("Avg Rent (4 Bed)", "avg_rent_4bd", None, "avg_rent_4bd", fmt_money)

    # Derived price metrics
    def compute_price_metrics_vals(asking, sqft, units, acres):
        pps = asking / sqft if isinstance(asking,(int,float)) and isinstance(sqft,(int,float)) and sqft else None
        ppu = asking / units if isinstance(asking,(int,float)) and isinstance(units,(int,float)) and units else None
        ppa = asking / acres if isinstance(asking,(int,float)) and isinstance(acres,(int,float)) and acres else None
        return pps, ppu, ppa

    om_pps, om_ppu, om_ppa = compute_price_metrics_vals(om.get("asking_price"), om.get("rentable_sqft"), om.get("total_units"), om.get("lot_size"))
    cx_pps, cx_ppu, cx_ppa = compute_price_metrics_vals(crexi.get("Asking Price"), crexi.get("SqFt"), crexi.get("Units"), crexi.get("Lot Size"))

    rows.append(["Price/SqFt", fmt_money(om_pps) if om_pps else None, fmt_money(cx_pps) if cx_pps else None, None])
    rows.append(["Price/Unit", fmt_money(om_ppu) if om_ppu else None, fmt_money(cx_ppu) if cx_ppu else None, None])
    rows.append(["Price/Acre", fmt_money(om_ppa) if om_ppa else None, fmt_money(cx_ppa) if cx_ppa else None, None])

    df = pd.DataFrame(rows, columns=["Metric", "OM", "Crexi", "Realtor"])
    return df

# ======================
# UI
# ======================
st.title("🏗️ CRE Scraping Demo (Puppet)")
st.caption("Property summary, rent table (Realtor comps), Financials, Vacancy & EGI, and an all-metrics rollup. PDF drag-and-drop; tolerances collapsed.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded_pdf = st.file_uploader("Drag & drop the OM **PDF** (optional)", type=["pdf"])
    st.markdown("**Note:** Demo doesn't parse the PDF; it uses sample OM values to show the flow.")
    with st.expander("Adjust rent tolerances", expanded=False):
        for k in RENT_KEYS:
            conf = RENT_METRICS[k]
            conf["tol_abs"] = st.number_input(f"{conf['label']} rent tol_abs ($)",
                                              value=float(conf["tol_abs"]), step=5.0, key=f"tol_abs_rent_{k}")

    run = st.button("Run Demo")

if run:
    om = parse_om(uploaded_pdf)
    crexi = fetch_crexi(query)
    realtor = fetch_realtor(query)

    # ---------- Property Summary ----------
    st.subheader("Property Summary")
    blurb = (
        "123 Main St is a stabilized, garden-style multifamily asset in a strong Tampa submarket with "
        "healthy renter demand and consistent absorption. Upside exists through modest unit upgrades and "
        "bringing OM rents in line with market benchmarks while maintaining durable occupancy."
    )
    cols = st.columns(3)
    summary_map = {
        "Address": om.get("address"),
        "OZ Status": om.get("oz_status"),
        "Cap Rate": fmt_percent(om.get("cap_rate")),
        "Rentable Sq Ft": f"{int(round(om.get('rentable_sqft'))):,}" if isinstance(om.get("rentable_sqft"), (int,float)) else "—",
        "Avg Sq Ft / Unit": f"{int(round(om.get('avg_sqft_per_type'))):,}" if isinstance(om.get("avg_sqft_per_type"), (int,float)) else "—",
        "Asking Price": fmt_money(om.get("asking_price")),
        "Year Built / Renovated": f"{int(round(om.get('year_built_or_renov')))}" if isinstance(om.get("year_built_or_renov"), (int,float)) else "—",
        "Lot Size (acres)": f"{om.get('lot_size'):.2f}" if isinstance(om.get("lot_size"), (int,float)) else "—",
        "Total Units": f"{int(round(om.get('total_units')))}" if isinstance(om.get("total_units"), (int,float)) else "—",
    }
    idx = 0
    for label, value in summary_map.items():
        with cols[idx % 3]:
            st.markdown(f"**{label}**")
            st.markdown(f"{value if value is not None else '—'}")
        idx += 1
    st.markdown(f"> {blurb}")

    # ---------- Rent Table (with property-wide totals row) ----------
    st.subheader("Unit Rents & GPR (OM vs Realtor)")
    rent_df_raw = build_rent_df(om, realtor)
    totals = compute_property_wide(rent_df_raw)
    st.dataframe(style_rent_df(rent_df_raw, totals), use_container_width=True, hide_index=True)

    # ---------- Financials ----------
    st.subheader("Financials")
    fin_df = build_financials_table(om, crexi)
    st.dataframe(style_financials_df(fin_df), use_container_width=True, hide_index=True)

    # ---------- Vacancy & EGI ----------
    st.subheader("Area Average Vacancy")
    st.markdown(f"**{fmt_percent(AREA_AVG_VACANCY)}**")

    # EGI = GPR × (1 − vacancy), using property-wide GPR on OM and Realtor rents
    gpr_om = totals["gpr_om"]
    gpr_r  = totals["gpr_r"]
    egi_om = gpr_om * (1 - AREA_AVG_VACANCY) if isinstance(gpr_om,(int,float)) else None
    egi_r  = gpr_r  * (1 - AREA_AVG_VACANCY) if isinstance(gpr_r,(int,float))  else None

    st.markdown(
        "EGI (Effective Gross Income) is calculated as **GPR × (1 − vacancy)** using the area average vacancy rate."
    )
    egi_rows = pd.DataFrame([
        ["GPR (OM rents)", fmt_money(gpr_om) if gpr_om else "—"],
        ["GPR (Realtor rents)", fmt_money(gpr_r) if gpr_r else "—"],
        ["EGI (OM rents)", fmt_money(egi_om) if egi_om else "—"],
        ["EGI (Realtor rents)", fmt_money(egi_r) if egi_r else "—"],
    ], columns=["Metric", "Value"])
    st.dataframe(egi_rows, use_container_width=True, hide_index=True)

    # ---------- All Metrics Rollup ----------
    st.subheader("All Metrics (OM, Crexi, Realtor)")
    all_df = build_all_metrics_table(om, crexi, realtor)
    st.dataframe(all_df, use_container_width=True, hide_index=True)

else:
    st.info("Drag & drop an OM **PDF** (optional), set a query, then click **Run Demo**.")
