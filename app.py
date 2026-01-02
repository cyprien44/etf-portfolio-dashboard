import io
import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pycountry
import unicodedata
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

ETF_BUCKET = {
    "IE0002XZSH01": "WORLD",    # iShares MSCI World Swap PEA
    "FR0013412038": "EUROPE",   # Amundi PEA MSCI Europe
    "FR0011869312": "ASIA",     # Amundi PEA Asie Pacifique
    "FR0013412020": "EM",       # Amundi PEA MSCI Emerging Markets ESG
    "FR0013412004": "LATAM",    # Amundi PEA MSCI Emerging Latin America
}

OVERLAP = {
    # MSCI World
    ("WORLD", "EUROPE"): 0.25,
    ("WORLD", "EM"): 0.10,
    ("WORLD", "ASIA"): 0.05,
    ("WORLD", "LATAM"): 0.03,

    # Europe
    ("EUROPE", "EM"): 0.05,
    ("EUROPE", "ASIA"): 0.00,
    ("EUROPE", "LATAM"): 0.00,

    # Emerging Markets
    ("EM", "ASIA"): 0.55,
    ("EM", "LATAM"): 0.05,

    # Sub EM
    ("ASIA", "LATAM"): 0.00,
}


# -----------------------------
# Country mapping to ISO3 (for map)
# -----------------------------
COUNTRY_TO_ISO3 = {
    # amériques / dev
    "etats unis": "USA",
    "royaume uni": "GBR",
    "japon": "JPN",
    "canada": "CAN",
    "suisse": "CHE",
    "france": "FRA",
    "allemagne": "DEU",
    "australie": "AUS",
    "pays bas": "NLD",
    "espagne": "ESP",
    "suede": "SWE",
    "italie": "ITA",
    "danemark": "DNK",
    "finlande": "FIN",
    "belgique": "BEL",
    "norvege": "NOR",
    "irlande": "IRL",
    "autriche": "AUT",
    "hongrie": "HUN",
    "portugal": "PRT",
    "pologne": "POL",
    "republique tcheque": "CZE",
    "grece": "GRC",

    # asie / em
    "chine": "CHN",
    "taiwan": "TWN",
    "inde": "IND",
    "coree du sud": "KOR",
    "malaisie": "MYS",
    "indonesie": "IDN",
    "thailande": "THA",
    "philippines": "PHL",
    "singapour": "SGP",
    "hong kong": "HKG",
    "nouvelle zelande": "NZL",
    "arabie saoudite": "SAU",
    "emirats arabes unis": "ARE",
    "qatar": "QAT",
    "koweit": "KWT",
    "turquie": "TUR",

    # afrique / latam
    "afrique du sud": "ZAF",
    "bresil": "BRA",
    "mexique": "MEX",
    "chili": "CHL",
    "perou": "PER",
    "colombie": "COL",

    # russie (si tu en as)
    "russie": "RUS",
}


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


from gspread.exceptions import APIError
import time

def write_tab(sh, tab_name: str, df: pd.DataFrame) -> None:
    ws = sh.worksheet(tab_name)

    values = [df.columns.tolist()] + df.astype(object).values.tolist()
    if not values:
        values = [["user", "isin", "weight"]]

    nrows = len(values)
    ncols = len(values[0])

    # A1 range (ok tant que <= 26 colonnes)
    end_col = chr(ord("A") + ncols - 1)
    rng = f"A1:{end_col}{nrows}"

    # retry light (quota / 5xx)
    for k in range(3):
        try:
            ws.update(rng, values)
            return
        except APIError as e:
            if k < 2:
                time.sleep(1.5 * (k + 1))
                continue
            # on affiche au lieu de re-crasher
            st.error("Erreur Google Sheets lors de l'écriture (update).")
            st.write("HTTP status:", getattr(e.response, "status_code", None))
            st.code(getattr(e.response, "text", str(e)), language="text")
            return

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
        text=d["exposure"].map(lambda v: f"{v:.2%}")  # affichage % sur les barres
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    height = max(520, 12 * len(d) + 160)  # ~15px par ligne
    fig.update_layout(
        yaxis_title="",
        xaxis_tickformat=".1%",
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=d["label"].tolist(),
        ticktext=d["label"].tolist(),
        automargin=True,
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


def _norm(s: str) -> str:
    s = (s or "").strip()
    # enlève accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # normalise tirets/espaces
    s = s.replace("-", "-").replace("–", "-")
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def country_to_iso3(label: str):
    if not isinstance(label, str) or not label.strip():
        return None

    s = _norm(label)
    if s in {"autres", "others", "autre"}:
        return None

    # 1) mapping direct iso3
    if s in COUNTRY_TO_ISO3:
        return COUNTRY_TO_ISO3[s]

    # 2) fallback pycountry (utile si un pays pas dans le dict)
    try:
        # on tente avec le label original + la version normalisée "title"
        for candidate in [label.strip(), s.title()]:
            c = pycountry.countries.search_fuzzy(candidate)[0]
            return c.alpha_3
    except Exception:
        return None



# -----------------------------
# Portfolio math
# -----------------------------

def parse_weight(x) -> float:
    """Accepte 0.2 / 0,2 / '20%' / '20,00%' / 20 (si % oublié) -> renvoie 0..1."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0

    s = str(x).strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if not s:
        return 0.0

    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]  # enlève %

    try:
        v = float(s)
    except Exception:
        return 0.0

    if is_pct:
        v /= 100.0

    # garde-fou propre
    return v if 0.0 <= v <= 1.0 else 0.0


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

def estimate_stocks_universe(meta_by_isin, weights, etf_bucket, overlap):
    """
    Estime le nombre d'actions effectivement exposées,
    en tenant compte :
    - des poids normalisés
    - des overlaps structurels entre indices
    """

    # 1️⃣ ETFs actifs
    active_isins = [isin for isin, w in weights.items() if w > 0]

    # 2️⃣ Somme pondérée brute
    total = 0.0
    for isin in active_isins:
        w = weights.get(isin, 0.0)
        n = meta_by_isin.get(isin, {}).get("Stocks number")
        if pd.notna(n):
            total += w * float(n)

    # 3️⃣ Correction d'overlap pondérée
    overlap_penalty = 0.0
    for i, isin_i in enumerate(active_isins):
        for isin_j in active_isins[i + 1:]:
            w_i = weights.get(isin_i, 0.0)
            w_j = weights.get(isin_j, 0.0)

            b_i = etf_bucket.get(isin_i)
            b_j = etf_bucket.get(isin_j)

            if not b_i or not b_j:
                continue

            key = (b_i, b_j)
            key_rev = (b_j, b_i)
            ov = overlap.get(key, overlap.get(key_rev, 0.0))

            if ov > 0:
                n_i = meta_by_isin.get(isin_i, {}).get("Stocks number", 0)
                n_j = meta_by_isin.get(isin_j, {}).get("Stocks number", 0)

                overlap_penalty += ov * min(n_i, n_j) * min(w_i, w_j)

    return max(total - overlap_penalty, 0)



def estimate_stocks_unique(meta_by_isin, weights, etf_bucket, overlap):
    """
    Estimation du nombre d'actions uniques
    - présence binaire (poids > 0 => ETF actif)
    - correction par overlap entre univers
    """

    # ETFs réellement présents
    active_isins = [isin for isin, w in weights.items() if w > 0]

    # base : somme brute
    total = 0.0
    for isin in active_isins:
        n = meta_by_isin.get(isin, {}).get("Stocks number")
        if pd.notna(n):
            total += float(n)

    # correction overlap pairwise
    overlap_penalty = 0.0
    for i, isin_i in enumerate(active_isins):
        for isin_j in active_isins[i+1:]:
            b_i = etf_bucket.get(isin_i)
            b_j = etf_bucket.get(isin_j)
            if not b_i or not b_j:
                continue

            key = (b_i, b_j)
            key_rev = (b_j, b_i)
            ov = overlap.get(key, overlap.get(key_rev, 0.0))

            if ov > 0:
                n_i = meta_by_isin.get(isin_i, {}).get("Stocks number", 0)
                n_j = meta_by_isin.get(isin_j, {}).get("Stocks number", 0)
                overlap_penalty += ov * min(n_i, n_j)

    return max(total - overlap_penalty, 0)

def effective_count_from_exposure(df: pd.DataFrame, col: str = "exposure") -> float:
    """N_eff = 1 / sum(w_i^2) sur une distribution de poids."""
    if df is None or df.empty or col not in df.columns:
        return 0.0
    w = df[col].astype(float).values
    s2 = (w ** 2).sum()
    return (1.0 / s2) if s2 > 0 else 0.0


def unique_count_from_exposure(df: pd.DataFrame, label_col: str = "label", col: str = "exposure") -> int:
    """Nombre de labels avec exposition > 0 (en excluant Autres)."""
    if df is None or df.empty:
        return 0
    d = df.copy()
    d[label_col] = d[label_col].astype(str)
    d = d[(d[col] > 0) & (~d[label_col].str.lower().isin(["autres", "others", "autre"]))]
    return int(d[label_col].nunique())

def neff_objective(w, isins, expos_by_isin, w0=None, lam=0.1):
    weights_tmp = normalize_weights(dict(zip(isins, w)))
    df_ctry_tmp = aggregate(expos_by_isin, weights_tmp, "country")
    neff = effective_count_from_exposure(df_ctry_tmp)

    penalty = 0.0
    if w0 is not None:
        penalty = lam * np.sum((w - w0) ** 2)

    return -(neff - penalty)



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
# Liste des users uniques (nettoyée)
users = (
    df_weights_all["user"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
    .tolist() #[::-1]
)

# fallback sécurité
if not users:
    users = ["cyprien"]

# Validate files_link
required = {"isin", "etf_name", "excel_url", "active", "Stocks number", "TER"}
missing = required - set(df_files.columns)
if missing:
    st.error(f"Onglet `{FILES_TAB}`: colonnes manquantes {sorted(missing)}")
    st.stop()

df_files["active"] = df_files["active"].astype(str).str.lower()
df_active = df_files[df_files["active"].isin(["1", "true", "yes", "y"])].copy()


# Stocks number -> int
df_active["Stocks number"] = pd.to_numeric(df_active["Stocks number"], errors="coerce")

# TER
df_active["TER"] = (
    df_active["TER"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
df_active["TER"] = pd.to_numeric(df_active["TER"], errors="coerce")  / 100.0

if df_active.empty:
    st.warning("Aucun ETF actif dans `files_link` (colonne active=1).")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("paramètres")
    user = st.selectbox(
        "utilisateur",
        options=sorted(users),
        index=users.index("cyprien") if "cyprien" in users else 0,
    )
    top_n = st.slider("top N", 5, 70, 30)
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
name_map = {str(r["isin"]).strip(): str(r["etf_name"]).strip() for _, r in df_active.iterrows()}
if not isins:
    st.error("Aucun ETF n'a pu être chargé. Vérifie les liens Drive et les onglets Amundi.")
    st.stop()

meta_by_isin = (
    df_active.set_index("isin")[["Stocks number", "TER"]]
    .to_dict(orient="index")
)

# Load user weights
df_user_w = df_weights_all[df_weights_all["user"].astype(str).str.lower() == user].copy() if not df_weights_all.empty else pd.DataFrame()
w_map = {}
if not df_user_w.empty:
    for _, rr in df_user_w.iterrows():
        isin = str(rr["isin"]).strip()
        w_map[isin] = parse_weight(rr.get("weight", 0.0))

# Weight sliders
with st.sidebar:
    st.subheader("Poids du portefeuille")
    weights_raw = {}

    default_equal = 1.0 / len(isins) if len(isins) else 0.0

    for isin in isins:
        etf_name = name_map.get(isin, isin)  # fallback = ISIN si pas trouvé
        weights_raw[isin] = st.slider(etf_name, 0.0, 1.0, float(w_map.get(isin, default_equal)), 0.01)

    weights = normalize_weights(weights_raw)

    # --- Mode focus 100% (graphes uniquement, sliders inchangés) ---
    st.markdown("---")
    st.caption("Mode focus (graphes uniquement)")

    focus_options = ["(aucun)"] + [f"{name_map.get(i,'')} — {i}" for i in isins]
    focus_choice = st.selectbox("Afficher 100% sur :", focus_options, index=0)

    focus_isin = None
    if focus_choice != "(aucun)":
        focus_isin = focus_choice.split(" — ")[-1].strip()


    st.caption("Poids normalisés (référence)")
    for isin, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        st.write(f"**{name_map.get(isin, '')}** — `{isin}` : **{w:.1%}**")

    st.caption(f"Somme normalisée = {sum(weights.values()):.2f}")

    if st.button("Save weights"):
        df_keep = (
            df_weights_all[df_weights_all["user"].astype(str).str.lower() != user].copy()
            if not df_weights_all.empty
            else pd.DataFrame(columns=["user", "isin", "weight"])
        )
        df_new = pd.DataFrame([
            {
                "user": user,
                "isin": k,
                "etf_name": name_map.get(k, ""), 
                "weight": float(v),
            }
            for k, v in weights.items()
        ])
        df_out = pd.concat([df_keep, df_new], ignore_index=True)

        # ✅ anti-NaN / anti-Inf (obligatoire pour JSON)
        df_out = df_out.replace([float("inf"), float("-inf")], None)
        df_out = df_out.where(pd.notna(df_out), None)
        
        write_tab(sh, WEIGHTS_TAB, df_out)
        
        st.success("sauvegardé ✅")

# Poids effectifs pour les graphes (focus si activé)
if "focus_isin" in locals() and focus_isin in isins:
    weights_effective = {i: (1.0 if i == focus_isin else 0.0) for i in isins}
else:
    weights_effective = weights

# --- stats calculées sur weights_effective (donc focus inclus) ---

# TER pondéré (focus-aware)
w_ter = 0.0
w_sum = 0.0
for isin, w in weights_effective.items():
    ter = meta_by_isin.get(isin, {}).get("TER")
    if ter is None or pd.isna(ter) or w <= 0:
        continue
    w_ter += w * float(ter)
    w_sum += w

ter_weighted = (w_ter / w_sum) if w_sum > 0 else None

# Actions (focus-aware)
stocks_universe = estimate_stocks_universe(
    meta_by_isin=meta_by_isin,
    weights=weights_effective,     # <-- IMPORTANT
    etf_bucket=ETF_BUCKET,
    overlap=OVERLAP,
)

stocks_unique_est = estimate_stocks_unique(
    meta_by_isin=meta_by_isin,
    weights=weights_effective,     # <-- IMPORTANT (sert à détecter w>0)
    etf_bucket=ETF_BUCKET,
    overlap=OVERLAP,
)


if "focus_isin" in locals() and focus_isin:
    st.info(f"Mode focus actif : **{name_map.get(focus_isin,'')}** ({focus_isin}) → graphes à 100% sur cet ETF")

# Aggregate
df_sector = aggregate(expos_by_isin, weights_effective, "sector")
df_curr = aggregate(expos_by_isin, weights_effective, "currency")
df_ctry = aggregate(expos_by_isin, weights_effective, "country")

countries_effective = effective_count_from_exposure(df_ctry, col="exposure")
countries_unique = unique_count_from_exposure(df_ctry, label_col="label", col="exposure")

st.markdown("### Résumé du portefeuille")

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.metric("TER moyen (pondéré)", f"{ter_weighted:.2%}" if ter_weighted is not None else "n/a")
with r1c2:
    st.metric("", "")  # vide (ou tu peux mettre autre chose plus tard)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.metric("Nombre d'actions pondérées (estim.)", f"{stocks_universe:,.0f}".replace(",", " "))
with r2c2:
    st.metric("Nombre d'actions uniques total", f"{stocks_unique_est:,.0f}".replace(",", " "))

r3c1, r3c2 = st.columns(2)
with r3c1:
    st.metric("Nombre de pays pondérés (estim.)", f"{countries_effective:.2f}")
with r3c2:
    st.metric("Nombre de pays uniques total", f"{countries_unique:d}")


# Charts
c1, c2, c3 = st.columns(3)

with c1:
    bar_chart_top(df_sector, "secteurs")

with c2:
    bar_chart_top(df_curr, "devises")

with c3:
    bar_chart_top(df_ctry, "pays")

missing = df_ctry.copy()
missing["iso3"] = missing["label"].map(country_to_iso3)
missing = missing[missing["iso3"].isna() & ~missing["label"].astype(str).str.lower().isin(["autres","others","autre"])]

if not missing.empty:
    st.warning("Pays non reconnus pour la carte (mapping à compléter) :")
    st.dataframe(missing[["label","exposure"]].sort_values("exposure", ascending=False), use_container_width=True)
    
# Map
st.subheader("Carte — exposition pays")
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
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>exposure: %{z:.2%}<extra></extra>"
    )
    fig.update_layout(
        height=700,  # 👈 plus gros
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Exposure", tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)

# OPTIMISATION------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("optimisation – maximiser le nombre effectif de pays")

if "opt_result" not in st.session_state:
    st.session_state["opt_result"] = None

if not focus_isin:
    if st.button("Optimiser N_eff (pays)", type="primary"):
        import numpy as np
        from scipy.optimize import minimize

        isins_opt = [i for i, w in weights.items() if w > 0]

        if len(isins_opt) < 2:
            st.session_state["opt_result"] = {
                "success": False,
                "message": "Il faut au moins 2 ETFs actifs."
            }
        else:
            x0 = np.array([weights[i] for i in isins_opt])
            x0 = x0 / x0.sum()

            bounds = [(0.0, 1.0)] * len(isins_opt)
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            res = minimize(
                neff_objective,
                x0=x0,
                args=(isins_opt, expos_by_isin),
                method="SLSQP",
                bounds=bounds,
                constraints=[cons],
                options={"maxiter": 300},
            )

            if not res.success:
                st.session_state["opt_result"] = {
                    "success": False,
                    "message": res.message
                }
            else:
                w_opt = res.x
                weights_opt = dict(zip(isins_opt, w_opt))

                df_before = aggregate(expos_by_isin, weights, "country")
                df_after  = aggregate(expos_by_isin, weights_opt, "country")

                neff_before = effective_count_from_exposure(df_before)
                neff_after  = effective_count_from_exposure(df_after)

                df_out = pd.DataFrame({
                    "ETF": [name_map.get(i, i) for i in isins_opt],
                    "poids actuel": [weights[i] for i in isins_opt],
                    "poids optimal": w_opt,
                })

                st.session_state["opt_result"] = {
                    "success": True,
                    "neff_before": float(neff_before),
                    "neff_after": float(neff_after),
                    "df_out": df_out,
                }

    # affichage persistant du dernier résultat
    opt = st.session_state.get("opt_result")
    if opt:
        if not opt["success"]:
            st.error(f"Optimisation échouée : {opt['message']}")
        else:
            st.write(f"**N_eff actuel** : {opt['neff_before']:.2f}")
            st.write(f"**N_eff optimal** : {opt['neff_after']:.2f}")

            df_show = opt["df_out"].copy()
            st.dataframe(
                df_show.assign(
                    **{
                        "poids actuel": df_show["poids actuel"].map(lambda x: f"{x:.2%}"),
                        "poids optimal": df_show["poids optimal"].map(lambda x: f"{x:.2%}"),
                    }
                ),
                use_container_width=True
            )

            if st.button("Effacer le résultat d'optimisation"):
                st.session_state["opt_result"] = None

else:
    st.info("Désactive le mode 100 % (focus) pour lancer l’optimisation.")


with st.expander("debug"):
    st.write("weights (normalisés)", pd.DataFrame([{"isin": k, "weight": v} for k, v in weights.items()]))
    st.write("countries", df_ctry)
    st.write("currencies", df_curr)
    st.write("sectors", df_sector)

