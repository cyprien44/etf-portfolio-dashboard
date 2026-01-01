import io
import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pycountry
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# Constants
# -----------------------------
FILES_TAB = "files_link"
WEIGHTS_TAB = "weights_users"

TAB_COUNTRY = "Répartition par pays"
TAB_CURRENCY = "Répartition par devise"
TAB_SECTOR = "Répartition par secteur"


# -----------------------------
# Google auth / Sheets
# -----------------------------
def get_gspread_client() -> gspread.Client:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)


def open_sheet(gc: gspread.Client):
    return gc.open_by_url(st.secrets["GOOGLE_SHEETS_URL"])


def read_tab(sh, tab_name: str) -> pd.DataFrame:
    ws = sh.worksheet(tab_name)
    records = ws.get_all_records()
    return pd.DataFrame(records)


def write_tab(sh, tab_name: str, df: pd.DataFrame) -> None:
    ws = sh.worksheet(tab_name)
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(object).values.tolist())


# -----------------------------
# Drive links → direct download
# -----------------------------
def drive_to_download_url(url: str) -> str:
    """
    Supports common share URLs:
    - https://drive.google.com/file/d/<ID>/view?usp=sharing
    - https://drive.google.com/open?id=<ID>
    - already direct links
    """
    if not url:
        return url

    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url

def bar_chart_top(df: pd.DataFrame, title: str):
    st.subheader(title)
    d = df.head(top_n).copy()
    # Pour avoir les plus gros en haut (Plotly met le 1er en bas en horizontal)
    d = d.sort_values("exposure", ascending=True)

    fig = px.bar(
        d,
        x="exposure",
        y="label",
        orientation="h",
        text=d["exposure"].map(lambda v: f"{v:.1%}")  # affichage % sur les barres
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="",
        xaxis_tickformat=".1%",
        margin=dict(l=10, r=10, t=30, b=10),
        height=500, 
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_excel_bytes(url: str) -> bytes:
    dl = drive_to_download_url(url)
    r = requests.get(dl, timeout=60)
    r.raise_for_status()
    return r.content


# -----------------------------
# Parsing Amundi tabs
# -----------------------------
def _to_float_pct(x) -> Optional[float]:
    """Return pct as fraction (0..1). Accepts '12,3%' or '12.3%' or 0.123 or 12.3."""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip().replace("\u00a0", " ").replace(",", ".")
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def parse_amundi_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust default for Amundi exports:
    often label is column B and weight is column C.
    We'll take 2nd+3rd columns when possible, otherwise 1st+2nd.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["label", "pct"])

    df = df_raw.copy()
    df = df.dropna(axis=1, how="all")  # drop empty columns
    cols = list(df.columns)
    if len(cols) >= 3:
        label_col, pct_col = cols[1], cols[2]
    elif len(cols) >= 2:
        label_col, pct_col = cols[0], cols[1]
    else:
        return pd.DataFrame(columns=["label", "pct"])

    out = df[[label_col, pct_col]].copy()
    out.columns = ["label", "pct"]
    out["label"] = out["label"].astype(str).str.strip()
    out["pct"] = out["pct"].map(_to_float_pct)

    out = out.dropna(subset=["pct"])
    out = out[~out["label"].str.lower().isin(["nan", "none", "", "total"])]
    out = out[out["pct"] > 0]

    # If sum looks like 100-ish, convert to 0..1
    if out["pct"].sum() > 1.5:
        out["pct"] = out["pct"] / 100.0

    out = out.groupby("label", as_index=False)["pct"].sum().sort_values("pct", ascending=False)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def read_exposures_from_excel(xlsx_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    needed = [TAB_COUNTRY, TAB_CURRENCY, TAB_SECTOR]
    missing = [t for t in needed if t not in xls.sheet_names]
    if missing:
        raise ValueError(f"Onglets manquants: {missing}. Onglets trouvés: {xls.sheet_names}")

    df_country = parse_amundi_table(pd.read_excel(xls, sheet_name=TAB_COUNTRY))
    df_curr = parse_amundi_table(pd.read_excel(xls, sheet_name=TAB_CURRENCY))
    df_sector = parse_amundi_table(pd.read_excel(xls, sheet_name=TAB_SECTOR))
    return df_country, df_curr, df_sector


# -----------------------------
# Country mapping to ISO3 (for map)
# -----------------------------
COUNTRY_FIX = {
    "États-Unis": "United States",
    "Etats-Unis": "United States",
    "Royaume-Uni": "United Kingdom",
    "Corée du Sud": "Korea, Republic of",
    "Corée du sud": "Korea, Republic of",
    "Russie": "Russian Federation",
    "Émirats arabes unis": "United Arab Emirates",
    "Emirats arabes unis": "United Arab Emirates",
    "République tchèque": "Czechia",
    "Republique tcheque": "Czechia",
    "Chine": "China",
    "Inde": "India",
    "Brésil": "Brazil",
    "Mexique": "Mexico",
}


def country_to_iso3(label: str) -> Optional[str]:
    if not isinstance(label, str) or not label.strip():
        return None
    s = label.strip()
    if s.lower() in ["autres", "others", "autre"]:
        return None
    s = COUNTRY_FIX.get(s, s)
    try:
        c = pycountry.countries.search_fuzzy(s)[0]
        return c.alpha_3
    except Exception:
        return None


# -----------------------------
# Portfolio math
# -----------------------------
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        return {k: 0.0 for k in w}
    return {k: max(0.0, float(v)) / total for k, v in w.items()}


def aggregate(expos_by_isin: Dict[str, Dict[str, pd.DataFrame]], weights: Dict[str, float], dim: str) -> pd.DataFrame:
    rows = []
    for isin, w in weights.items():
        if isin not in expos_by_isin:
            continue
        df = expos_by_isin[isin][dim].copy()
        df["exposure"] = df["pct"] * w
        rows.append(df[["label", "exposure"]])
    if not rows:
        return pd.DataFrame(columns=["label", "exposure"])
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby("label", as_index=False)["exposure"].sum().sort_values("exposure", ascending=False)
    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ETF dashboard", layout="wide")
st.title("ETF dashboard — expositions pays / devises / secteurs")

gc = get_gspread_client()
sh = open_sheet(gc)

df_files = read_tab(sh, FILES_TAB)
df_weights_all = read_tab(sh, WEIGHTS_TAB) if WEIGHTS_TAB in [w.title for w in sh.worksheets()] else pd.DataFrame(
    columns=["user", "isin", "weight"]
)

# Validate files_link
required = {"isin", "etf_name", "excel_url", "active"}
missing = required - set(df_files.columns)
if missing:
    st.error(f"Onglet `{FILES_TAB}`: colonnes manquantes {sorted(missing)}")
    st.stop()

df_files["active"] = df_files["active"].astype(str).str.lower()
df_active = df_files[df_files["active"].isin(["1", "true", "yes", "y"])].copy()

if df_active.empty:
    st.warning("Aucun ETF actif dans `files_link` (colonne active=1).")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("paramètres")
    user = st.text_input("user", value="cyprien").strip().lower()
    top_n = st.slider("top N", 5, 50, 15)
    if st.button("recharger les excels"):
        # Clear caches (download + parsing) if you updated files in Drive
        st.cache_data.clear()
        st.success("cache vidé. rafraîchis la page.")

# Load & parse all active ETFs
expos_by_isin: Dict[str, Dict[str, pd.DataFrame]] = {}
errors: List[Tuple[str, str]] = []

with st.spinner("Chargement des excels (cache 1h) ..."):
    for _, r in df_active.iterrows():
        isin = str(r["isin"]).strip()
        url = str(r["excel_url"]).strip()
        try:
            b = fetch_excel_bytes(url)
            df_country, df_curr, df_sector = read_exposures_from_excel(b)
            expos_by_isin[isin] = {"country": df_country, "currency": df_curr, "sector": df_sector}
        except Exception as e:
            errors.append((isin, str(e)))

if errors:
    st.warning("Certains ETFs n'ont pas pu être lus :")
    st.dataframe(pd.DataFrame(errors, columns=["isin", "error"]), use_container_width=True)

isins = sorted(expos_by_isin.keys())
#name_map = {str(r["isin"]).strip(): str(r["etf_name"]).strip() for _, r in df_active.iterrows()}
#name_map = { str(r["etf_name"]).strip() for _, r in df_active.iterrows()}
if not isins:
    st.error("Aucun ETF n'a pu être chargé. Vérifie les liens Drive et les onglets Amundi.")
    st.stop()

# Load user weights
df_user_w = df_weights_all[df_weights_all["user"].astype(str).str.lower() == user].copy() if not df_weights_all.empty else pd.DataFrame()
w_map = {}
if not df_user_w.empty:
    for _, rr in df_user_w.iterrows():
        try:
            w_map[str(rr["isin"]).strip()] = float(rr["weight"])
        except Exception:
            pass

# Weight sliders
with st.sidebar:
    st.subheader("poids du portefeuille")
    weights_raw = {}

    default_equal = 1.0 / len(isins) if len(isins) else 0.0

    for isin in isins:
        label = f"{isin} — {name_map.get(isin, '')}".strip()
        weights_raw[isin] = st.slider(label, 0.0, 1.0, float(w_map.get(isin, default_equal)), 0.01)

    weights = normalize_weights(weights_raw)

    # Afficher les % normalisés à côté (liste triée)
    st.caption("poids normalisés")
    for isin, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        st.write(f"**{isin}** — {name_map.get(isin, '')} : **{w:.1%}**")

    st.caption(f"somme normalisée = {sum(weights.values()):.2f}")

    if st.button("save weights"):
        df_keep = (
            df_weights_all[df_weights_all["user"].astype(str).str.lower() != user].copy()
            if not df_weights_all.empty
            else pd.DataFrame(columns=["user", "isin", "weight"])
        )
        df_new = pd.DataFrame([{"user": user, "isin": k, "weight": v} for k, v in weights.items()])
        df_out = pd.concat([df_keep, df_new], ignore_index=True)
        write_tab(sh, WEIGHTS_TAB, df_out)
        st.success("sauvegardé ✅")


# Aggregate
df_sector = aggregate(expos_by_isin, weights, "sector")
df_curr = aggregate(expos_by_isin, weights, "currency")
df_ctry = aggregate(expos_by_isin, weights, "country")

# Charts
c1, c2, c3 = st.columns(3)

with c1:
    bar_chart_top(df_sector, "secteurs")

with c2:
    bar_chart_top(df_curr, "devises")

with c3:
    bar_chart_top(df_ctry, "pays")


# Map
st.subheader("carte — exposition pays")
df_map = df_ctry.copy()
df_map["iso3"] = df_map["label"].map(country_to_iso3)
df_map = df_map.dropna(subset=["iso3"])

if df_map.empty:
    st.info("Impossible d'afficher la carte (pays non mappés). On pourra enrichir le mapping ensuite.")
else:
    fig = px.choropleth(
        df_map,
        locations="iso3",
        color="exposure",
        hover_name="label",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=700,  # 👈 plus gros
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Exposure", tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("debug"):
    st.write("weights (normalisés)", pd.DataFrame([{"isin": k, "weight": v} for k, v in weights.items()]))
    st.write("countries", df_ctry)
    st.write("currencies", df_curr)
    st.write("sectors", df_sector)
