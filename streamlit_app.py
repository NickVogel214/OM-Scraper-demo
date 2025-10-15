import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        "units_1bd": 32, "units_2bd": 48, "units_3bd": 30, "units_4bd": 10,
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
        "avg_rent_1bd": 1525, "avg_rent_2bd": 1935,
        "avg_rent_3bd": 2260, "avg_rent_4bd": 2550,
        "units_1bd": 29, "units_2bd": 49,
        "total_units": 118,
        "asking_price": 33_500_000,
        # realtor often lacks NOI / sqft; that's fine for the demo
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

# Location Data — only these compared (vs Crexi only)
LOC_METRICS: List[Tuple[str, Dict[str, Any]]] = [
    ("total_units",         {"label": "Total Units",              "type": "num", "tol_abs": 2}),
    ("lot_size",            {"label": "Lot Size (acres)",         "type": "num", "tol_rel": 0.10}),
    ("year_built_or_renov", {"label": "Year Built / Renovated",   "type": "year"}),  # special
    ("rentable_sqft",       {"label": "Rentable Sq. Ft.",         "type": "num", "tol_rel": 0.05}),
]

NO_COMMA_KEYS = {"year_built_or_renov"}

# Tolerances for plot spreads
TOL_AVG_SQFT_ABS = 50.0  # vs Realtor
TOL_NOI_REL = 0.05       # vs Crexi
TOL_PRICE_REL = 0.03     # vs Crexi

# Mock: % of similar properties that are OZ (puppet number)
MOCK_OZ_MARKET_PCT = 0.22

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

def year_age_category(year_val: Optional[float]) -> str:
    if not isinstance(year_val, (int, float)):
        return "—"
    y = int(year_val)
    current_year = datetime.now().year
    age = max(0, current_year - y)
    if age <= 5: return "Very New"
    if age <= 15: return "New"
    if age <= 30: return "Standard"
    if age <= 50: return "Old"
    return "Very Old"

# ======================
# Rent table (Units # | Avg Rent | GPR) with Realtor value + signed OM vs avg Δ%
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

        thresh = threshold_pct_vs_ref(tol_abs, None, avg_rent)

        # GPRs (annual) using OM units always
        gpr_om = gpr_r = None
        if isinstance(om_units, (int, float)):
            if isinstance(om_rent, (int, float)):
                gpr_om = om_units * om_rent * 12
            if isinstance(r_rent, (int, float)):
                gpr_r  = om_units * r_rent  * 12

        rows.append({
            "Unit Type": label,
            "Units #": int(om_units) if isinstance(om_units, (int, float)) else om_units,
            # Avg Rent shows Realtor cash + signed OM vs avg %
            "Avg Rent": (r_rent, dev_vs_avg, avg_rent, tol_abs),
            "GPR": (gpr_om, gpr_r),
        })
    return pd.DataFrame(rows)

def style_rent_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format units
    show["Units #"] = show["Units #"].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)

    # Format Avg Rent content with sign
    def fmt_avg_rent(cell):
        if not isinstance(cell, tuple) or len(cell) < 4:
            return ""
        r_rent, dev_vs_avg, avg_rent, tol_abs = cell
        r_str = fmt_money(r_rent) if isinstance(r_rent, (int, float)) else "—"
        d_str = signed_pct(dev_vs_avg)
        return f"R: {r_str} | OM vs avg: {d_str}"

    show["Avg Rent"] = show["Avg Rent"].apply(fmt_avg_rent)

    # Format GPR
    def fmt_gpr(pair):
        if not isinstance(pair, tuple):
            return ""
        om, rr = pair
        om_s = fmt_money(om) if isinstance(om, (int, float)) else "—"
        rr_s = fmt_money(rr) if isinstance(rr, (int, float)) else "—"
        return f"OM: {om_s} | R: {rr_s}"

    show["GPR"] = show["GPR"].apply(fmt_gpr)

    styler = show[["Unit Type", "Units #", "Avg Rent", "GPR"]].style

    # Left color bar on Avg Rent cell based on OM vs avg Δ% (signed)
    def left_bar_styles():
        styles = []
        for _, row in df.iterrows():
            r_rent, dev_vs_avg, avg_rent, tol_abs = row["Avg Rent"]
            thresh = threshold_pct_vs_ref(tol_abs, None, avg_rent)
            col = color_by_dev(dev_vs_avg, thresh)
            if col:
                styles.append(f"border-left: 8px solid {col}; padding-left: 6px;")
            else:
                styles.append("")
        return styles

    styler = styler.apply(lambda _: left_bar_styles(), axis=0, subset=["Avg Rent"])

    # Zebra
    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# Location data table (selected metrics; compare vs Crexi only)
# Year built shows an age category instead of Δ%
# ======================
def build_location_df(om: Dict[str, Any], crexi: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, rule in LOC_METRICS:
        label = rule["label"]
        typ = rule["type"]

        o = om.get(key)
        c = crexi.get(key)

        if typ == "year":
            category = year_age_category(o)
            rows.append({
                "Metric": label,
                "OM": o,
                "Crexi": c,
                "Δ vs Crexi / Category": category,
                "_key": key,
                "_type": typ,
            })
        else:
            tol_abs = rule.get("tol_abs", None)
            tol_rel = rule.get("tol_rel", None)
            dev_c = pct_dev_from_ref(o, c)
            thresh_c = threshold_pct_vs_ref(tol_abs, tol_rel, c)
            rows.append({
                "Metric": label,
                "OM": o,
                "Crexi": c,
                "Δ vs Crexi / Category": (dev_c, thresh_c),
                "_key": key,
                "_type": typ,
            })
    return pd.DataFrame(rows)

def style_location_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format values and Δ / Category
    def fmt_delta(cell):
        if isinstance(cell, tuple) and len(cell) == 2:
            dev, _ = cell
            return signed_pct(dev)
        return cell  # category string

    for i, row in df.iterrows():
        key = row["_key"]
        show.at[i, "OM"] = fmt_number(key, row["OM"])
        show.at[i, "Crexi"] = fmt_number(key, row["Crexi"])
        show.at[i, "Δ vs Crexi / Category"] = fmt_delta(row["Δ vs Crexi / Category"])

    styler = show[["Metric", "OM", "Crexi", "Δ vs Crexi / Category"]].style

    # Color numeric Δ% cells
    def colors_for_delta():
        out = []
        for _, row in df.iterrows():
            cell = row["Δ vs Crexi / Category"]
            raw = df.loc[_, "Δ vs Crexi / Category"]
            if isinstance(raw, tuple) and len(raw) == 2:
                dev, thr = raw
                out.append(f"background-color: {color_by_dev(dev, thr)};")
            else:
                out.append("")  # year category
        return out

    styler = styler.apply(lambda _: colors_for_delta(), axis=0, subset=["Δ vs Crexi / Category"])

    styler = styler.set_properties(**{"border-color": "#ddd"}) \
                   .set_table_styles([{"selector": "tbody tr:nth-child(even)",
                                       "props": [("background-color", "#fafafa")]}])
    return styler

# ======================
# Box plots (synthetic) sections
# ======================
def _synthetic_samples(mu: float, sigma: float, n: int = 500) -> np.ndarray:
    return np.random.normal(loc=float(mu), scale=max(float(sigma), 1e-6), size=n)

def boxplot_single_series(data: np.ndarray, label: str, title: str, y_label: str) -> Optional[plt.Figure]:
    if data.size == 0:
        return None
    fig = plt.figure()
    plt.boxplot([data], labels=[label], showmeans=True)
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    return fig

def section_boxplot_avg_sqft(om: Dict[str, Any], realtor: Dict[str, Any]) -> Optional[plt.Figure]:
    r = realtor.get("avg_sqft_per_type")
    if not isinstance(r, (int, float)):
        return None
    samples = _synthetic_samples(r, TOL_AVG_SQFT_ABS)
    return boxplot_single_series(samples, "Realtor", "Avg Sq Ft / Unit (Realtor-only)", "Sq Ft")

def section_boxplot_price(om: Dict[str, Any], crexi: Dict[str, Any]) -> Optional[plt.Figure]:
    c = crexi.get("asking_price")
    if not isinstance(c, (int, float)):
        return None
    sigma = abs(c) * TOL_PRICE_REL
    samples = _synthetic_samples(c, sigma)
    return boxplot_single_series(samples, "Crexi", "Asking Price (Crexi-only)", "Price ($)")

def section_boxplot_noi(om: Dict[str, Any], crexi: Dict[str, Any]) -> Optional[plt.Figure]:
    c = crexi.get("noi")
    if not isinstance(c, (int, float)):
        return None
    sigma = abs(c) * TOL_NOI_REL
    samples = _synthetic_samples(c, sigma)
    return boxplot_single_series(samples, "Crexi", "NOI (Crexi-only)", "NOI ($)")

# ======================
# CSV export (Metric, OM, Crexi, Realtor, Recommended)
# ======================
def recommended_value(metric: str, om_val, crexi_val, realtor_val) -> Optional[float]:
    """
    Rule of thumb:
    - Rents & Avg Sq Ft -> lean on Realtor (0.7*Realtor + 0.3*OM)
    - Asking Price & NOI -> lean on Crexi (0.7*Crexi + 0.3*OM)
    - Else -> if Crexi present use (0.6*Crexi + 0.4*OM), elif Realtor present use (0.6*Realtor + 0.4*OM), else OM.
    """
    def wblend(ref, om, w=0.7):
        if isinstance(ref, (int, float)) and isinstance(om, (int, float)):
            return w*ref + (1-w)*om
        return ref if isinstance(ref, (int, float)) else om if isinstance(om, (int, float)) else None

    if metric in ["avg_rent_1bd","avg_rent_2bd","avg_rent_3bd","avg_rent_4bd","avg_sqft_per_type"]:
        return wblend(realtor_val, om_val, 0.7)
    if metric in ["asking_price","noi"]:
        return wblend(crexi_val, om_val, 0.7)
    # fallback
    if isinstance(crexi_val, (int,float)):
        return wblend(crexi_val, om_val, 0.6)
    if isinstance(realtor_val, (int,float)):
        return wblend(realtor_val, om_val, 0.6)
    return om_val if isinstance(om_val, (int,float)) else None

def build_export_rows(om: Dict[str, Any], crexi: Dict[str, Any], realtor: Dict[str, Any]) -> pd.DataFrame:
    metrics = [
        # Rents
        "avg_rent_1bd","avg_rent_2bd","avg_rent_3bd","avg_rent_4bd",
        # Location summary & key comps
        "avg_sqft_per_type","total_units","lot_size","year_built_or_renov","rentable_sqft",
        # Financials
        "asking_price","noi","cap_rate","expense_ratio"
    ]
    labels = {
        "avg_rent_1bd":"Avg Rent (1 Bed)","avg_rent_2bd":"Avg Rent (2 Bed)",
        "avg_rent_3bd":"Avg Rent (3 Bed)","avg_rent_4bd":"Avg Rent (4 Bed)",
        "avg_sqft_per_type":"Avg Sq Ft / Unit","total_units":"Total Units","lot_size":"Lot Size (acres)",
        "year_built_or_renov":"Year Built / Renovated","rentable_sqft":"Rentable Sq Ft",
        "asking_price":"Asking Price","noi":"NOI","cap_rate":"Cap Rate","expense_ratio":"Expense Ratio"
    }
    rows = []
    for m in metrics:
        rec = recommended_value(m, om.get(m), crexi.get(m), realtor.get(m))
        rows.append({
            "Metric": labels[m],
            "OM": om.get(m),
            "Crexi": crexi.get(m),
            "Realtor": realtor.get(m),
            "Recommended": rec
        })
    return pd.DataFrame(rows)

def format_export_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, row in out.iterrows():
        key = None
        # find metric key by reverse map (not stored); infer from label map for formatting simplicity
        # we'll format by guessing types based on label
        label = row["Metric"]
        val_cols = ["OM","Crexi","Realtor","Recommended"]
        def fmt_auto(lbl, v):
            if v is None or not isinstance(v, (int,float)): return v
            if "Price" in lbl or lbl in ["NOI"]:
                return fmt_money(v)
            if "Cap Rate" in lbl or "Expense Ratio" in lbl:
                return fmt_percent(v)
            if "Year" in lbl:
                return f"{int(v)}"
            if "Sq Ft" in lbl:
                return fmt_number("generic", v)
            if "Rent" in lbl:
                return fmt_money(v)
            if "Units" in lbl:
                return f"{int(v)}"
            if "Lot Size" in lbl:
                return fmt_number("generic", v)
            return fmt_number("generic", v)
        for c in val_cols:
            out.at[i, c] = fmt_auto(label, row[c])
    return out

# ======================
# UI
# ======================
st.title("🏗️ CRE Scraping Demo (Puppet)")
st.caption("Property summary + rent deviations (Realtor) & selected location comps (vs Crexi). Plus OZ market stat and three box-plot sections.")

with st.sidebar:
    st.header("Controls")
    query = st.text_input("Search query / address", "123 Main St, Tampa, FL")
    uploaded_pdf = st.file_uploader("Drag & drop the OM **PDF** (optional)", type=["pdf"])
    st.markdown("**Note:** Demo doesn't parse the PDF; it uses sample OM values to show the flow.")
    with st.expander("Adjust tolerances", expanded=False):
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

    # OZ market stat (mock)
    st.subheader("Market OZ Share")
    st.write(f"Estimated **{fmt_percent(MOCK_OZ_MARKET_PCT)}** of comparable assets are located in Opportunity Zones.")
    st.progress(min(1.0, max(0.0, MOCK_OZ_MARKET_PCT)))

    # ---------- Rent Table ----------
    st.subheader("Unit Rents & GPR")
    rent_df_raw = build_rent_df(om, realtor)
    st.dataframe(style_rent_df(rent_df_raw), use_container_width=True, hide_index=True)

    # ---------- Box Plot: Avg Sq Ft / Unit (Realtor) ----------
    st.subheader("Avg Sq Ft / Unit — Realtor Box Plot")
    fig_sqft = section_boxplot_avg_sqft(om, realtor)
    if fig_sqft is not None:
        st.pyplot(fig_sqft)

    # ---------- Box Plot: Asking Price (Crexi) ----------
    st.subheader("Asking Price — Crexi Box Plot")
    fig_price = section_boxplot_price(om, crexi)
    if fig_price is not None:
        st.pyplot(fig_price)

    # ---------- Box Plot: NOI (Crexi) ----------
    st.subheader("NOI — Crexi Box Plot")
    fig_noi = section_boxplot_noi(om, crexi)
    if fig_noi is not None:
        st.pyplot(fig_noi)

    # ---------- Location Data (Selected vs Crexi) ----------
    st.subheader("Location Data (Selected Comparisons vs Crexi)")
    loc_df_raw = build_location_df(om, crexi)
    st.dataframe(style_location_df(loc_df_raw), use_container_width=True, hide_index=True)

    # ---------- CSV Export ----------
    st.subheader("Download Comparison CSV")
    export_df = build_export_rows(om, crexi, realtor)
    pretty_export = format_export_df(export_df)
    csv_bytes = pretty_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (OM, Crexi, Realtor, Recommended)",
                       data=csv_bytes, file_name="compare_export.csv", mime="text/csv")

else:
    st.info("Drag & drop an OM **PDF** (optional), set a query, then click **Run Demo**.")
