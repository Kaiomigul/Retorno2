import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# Plotly
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

# =========================
# CORES
# =========================
SELIC_COLOR = "#c00000"

CARTEIRA_COLOR = {
    "Conservador": "#08082a",
    "Moderado": "#c00000",
    "Agressivo": "#dfac16",
}

CDI_BASE100_COLOR = "#444444"  # CDI no gráfico Base100

st.set_page_config(page_title="Retorno Esperado", layout="wide")
st.title("Retorno Esperado — Classes e Carteiras (condicional ao CDI)")

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dados.xlsx"

assets = ['RV Brasil', 'CDI', 'IRFM', 'IMAB', 'IDA', 'AGG Hedge', 'SPY Hedge', 'AGG', 'SPY']

portfolios = {
    'Conservador': {'RV Brasil': 0.00, 'CDI': 0.20, 'IRFM': 0.05, 'IMAB': 0.15, 'IDA': 0.55, 'AGG Hedge': 0.00, 'SPY Hedge': 0.00, 'AGG': 0.05, 'SPY': 0.00},
    'Moderado':    {'RV Brasil': 0.10, 'CDI': 0.10, 'IRFM': 0.10, 'IMAB': 0.25, 'IDA': 0.30, 'AGG Hedge': 0.00, 'SPY Hedge': 0.00, 'AGG': 0.05, 'SPY': 0.10},
    'Agressivo':   {'RV Brasil': 0.20, 'CDI': 0.05, 'IRFM': 0.10, 'IMAB': 0.25, 'IDA': 0.20, 'AGG Hedge': 0.00, 'SPY Hedge': 0.00, 'AGG': 0.05, 'SPY': 0.15},
}
carteira_order = ["Conservador", "Moderado", "Agressivo"]
WINDOW_ANUAL = 252

# =========================
# FUNÇÕES
# =========================
def rolling_annual_factor(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.exp(np.log1p(s).rolling(window=window).sum())

def calculate_metrics(factor_series: pd.Series, z: float):
    s = pd.to_numeric(factor_series, errors="coerce").dropna()
    if s.empty:
        return None
    mu = float(s.mean())
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    lower = mu - z * sd
    upper = mu + z * sd
    return mu, lower, upper, int(len(s))

def compute_everything(df_raw: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df_raw.copy()

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    df[assets] = df[assets].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df["SelicAnualizada"] = (1.0 + df["CDI"]).pow(window) - 1.0
    df["SelicTrend"] = np.where(df["SelicAnualizada"].diff() >= 0, "Subindo", "Caindo")

    for col in assets:
        df[f"Base100_{col}"] = 100 * (1 + df[col]).cumprod()

    for name, wdict in portfolios.items():
        w = np.array([wdict.get(a, 0.0) for a in assets], dtype=float)
        df[f"Retorno_{name}"] = df[assets].values @ w
        df[f"Base100_{name}"] = 100 * (1 + df[f"Retorno_{name}"]).cumprod()
        df[f"FatorAnual_{name}"] = rolling_annual_factor(df[f"Retorno_{name}"], window=window)

    df["CDIAcumuladoAnual"] = rolling_annual_factor(df["CDI"], window=window)
    return df

def build_cdi_regime_table(df: pd.DataFrame, bins: np.ndarray, z: float) -> pd.DataFrame:
    results = []

    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])

        lo_pct = (lo - 1.0) * 100.0
        hi_pct = (hi - 1.0) * 100.0
        range_label_pct = f"{lo_pct:.0f}% - {hi_pct:.0f}%"

        cdi_mid = (lo + hi) / 2.0
        cdi_mid_pct = (cdi_mid - 1.0) * 100.0

        filtered = df[(df["CDIAcumuladoAnual"] >= lo) & (df["CDIAcumuladoAnual"] < hi)].copy()

        for name in carteira_order:
            col = f"FatorAnual_{name}"
            metrics = calculate_metrics(filtered[col], z=z)
            if metrics is None:
                continue

            exp_factor, lower_factor, upper_factor, nobs = metrics

            exp_pct = (exp_factor - 1.0) * 100.0
            lower_pct = (lower_factor - 1.0) * 100.0
            upper_pct = (upper_factor - 1.0) * 100.0

            if cdi_mid_pct != 0:
                exp_pct_cdi = (exp_pct / cdi_mid_pct) * 100.0
                lower_pct_cdi = (lower_pct / cdi_mid_pct) * 100.0
                upper_pct_cdi = (upper_pct / cdi_mid_pct) * 100.0
            else:
                exp_pct_cdi = lower_pct_cdi = upper_pct_cdi = np.nan

            results.append({
                "CDI Range (% a.a.)": range_label_pct,
                "_range_order": i,
                "CDI Midpoint (% a.a.)": cdi_mid_pct,

                "Carteira": name,

                "Expected Return (% a.a.)": exp_pct,
                "Lower Bound (% a.a.)": lower_pct,
                "Upper Bound (% a.a.)": upper_pct,

                "Expected Return (% do CDI)_base": exp_pct_cdi,
                "Lower Bound (% do CDI)_base": lower_pct_cdi,
                "Upper Bound (% do CDI)_base": upper_pct_cdi,

                "Observações": nobs,
            })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    num_cols = [
        "CDI Midpoint (% a.a.)",
        "Expected Return (% a.a.)", "Lower Bound (% a.a.)", "Upper Bound (% a.a.)",
        "Expected Return (% do CDI)_base", "Lower Bound (% do CDI)_base", "Upper Bound (% do CDI)_base",
    ]
    for c in num_cols:
        res[c] = pd.to_numeric(res[c], errors="coerce")

    res["Carteira"] = pd.Categorical(res["Carteira"], categories=carteira_order, ordered=True)
    res = res.sort_values(["_range_order", "Carteira"]).reset_index(drop=True)
    return res

def smooth_by_portfolio(df: pd.DataFrame, col: str, window_size: int) -> pd.Series:
    return (
        df.groupby("Carteira", sort=False)[col]
          .apply(lambda s: s.rolling(window=window_size, min_periods=1, center=True).mean())
          .reset_index(level=0, drop=True)
    )

def prepare_visual_table_consistent(res_base: pd.DataFrame, vis_tipo: str, vis_w: int) -> pd.DataFrame:
    """
    CONSISTÊNCIA:
    - Fonte: retorno % a.a. (base ou suavizado)
    - %CDI é sempre derivado do retorno %a.a / CDI_midpoint
    """
    df = res_base.copy()

    if vis_tipo == "Suavizado":
        df["VIS Expected Return (% a.a.)"] = smooth_by_portfolio(df, "Expected Return (% a.a.)", vis_w)
        df["VIS Lower Bound (% a.a.)"] = smooth_by_portfolio(df, "Lower Bound (% a.a.)", vis_w)
        df["VIS Upper Bound (% a.a.)"] = smooth_by_portfolio(df, "Upper Bound (% a.a.)", vis_w)
    else:
        df["VIS Expected Return (% a.a.)"] = df["Expected Return (% a.a.)"]
        df["VIS Lower Bound (% a.a.)"] = df["Lower Bound (% a.a.)"]
        df["VIS Upper Bound (% a.a.)"] = df["Upper Bound (% a.a.)"]

    cdi_mid = pd.to_numeric(df["CDI Midpoint (% a.a.)"], errors="coerce").replace(0, np.nan)

    df["VIS Expected Return (% do CDI)"] = (df["VIS Expected Return (% a.a.)"] / cdi_mid) * 100.0
    df["VIS Lower Bound (% do CDI)"] = (df["VIS Lower Bound (% a.a.)"] / cdi_mid) * 100.0
    df["VIS Upper Bound (% do CDI)"] = (df["VIS Upper Bound (% a.a.)"] / cdi_mid) * 100.0

    return df

def make_grouped_bar_chart(bar_long: pd.DataFrame, y_field: str, y_title: str) -> alt.Chart:
    color_scale = alt.Scale(domain=carteira_order, range=[CARTEIRA_COLOR[c] for c in carteira_order])

    bars = (
        alt.Chart(bar_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "cdi_range:N",
                sort=alt.SortField(field="range_order", order="ascending"),
                title="Faixa do CDI (% a.a.)",
            ),
            xOffset=alt.XOffset("portfolio:N", sort=carteira_order),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color("portfolio:N", scale=color_scale, legend=alt.Legend(title="Carteira")),
            tooltip=[
                alt.Tooltip("cdi_range:N", title="Faixa CDI"),
                alt.Tooltip("portfolio:N", title="Carteira"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=420)
    )

    labels = (
        alt.Chart(bar_long)
        .mark_text(dy=-6, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("cdi_range:N", sort=alt.SortField(field="range_order", order="ascending")),
            xOffset=alt.XOffset("portfolio:N", sort=carteira_order),
            y=alt.Y(f"{y_field}:Q"),
            text=alt.Text(f"{y_field}:Q", format=".1f"),
        )
    )

    return bars + labels

def make_dual_axis_selic_base100_with_cdi(df: pd.DataFrame) -> alt.Chart:
    base_cols = [f"Base100_{n}" for n in carteira_order] + ["Base100_CDI"]
    tmp = df[base_cols].copy()
    tmp.columns = carteira_order + ["CDI"]
    long_base = tmp.reset_index().melt(
        id_vars=[tmp.reset_index().columns[0]],
        var_name="serie",
        value_name="base100"
    )
    date_col = long_base.columns[0]
    long_base = long_base.rename(columns={date_col: "data"})

    selic_df = df[["SelicAnualizada"]].copy()
    selic_df = selic_df.reset_index().rename(columns={selic_df.reset_index().columns[0]: "data"})
    selic_df["selic_aa_pct"] = selic_df["SelicAnualizada"] * 100.0

    domain = carteira_order + ["CDI"]
    range_colors = [CARTEIRA_COLOR[c] for c in carteira_order] + [CDI_BASE100_COLOR]
    series_color = alt.Scale(domain=domain, range=range_colors)

    base_lines = (
        alt.Chart(long_base)
        .mark_line()
        .encode(
            x=alt.X("data:T", title="Data"),
            y=alt.Y("base100:Q", title="Base 100"),
            color=alt.Color("serie:N", scale=series_color, legend=alt.Legend(title="Série")),
            strokeDash=alt.condition(
                alt.datum.serie == "CDI",
                alt.value([6, 4]),
                alt.value([1, 0])
            ),
            tooltip=[
                alt.Tooltip("data:T", title="Data"),
                alt.Tooltip("serie:N", title="Série"),
                alt.Tooltip("base100:Q", title="Base 100", format=".2f"),
            ],
        )
    )

    selic_line = (
        alt.Chart(selic_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("data:T"),
            y=alt.Y("selic_aa_pct:Q", title="Selic anualizada (% a.a.)", axis=alt.Axis(orient="right")),
            color=alt.value(SELIC_COLOR),
            tooltip=[
                alt.Tooltip("data:T", title="Data"),
                alt.Tooltip("selic_aa_pct:Q", title="Selic anualizada (% a.a.)", format=".2f"),
            ],
        )
    )

    return (
        alt.layer(base_lines, selic_line)
        .resolve_scale(y="independent")
        .properties(height=520, title="Base 100 (Carteiras + CDI) e Selic anualizada (eixo direito)")
    )

# --- helper robusto: prepara x,y sem NaN e sem duplicatas ---
def _prepare_xy_unique(sub: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    tmp = sub[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])

    # remove duplicatas de x (se acontecer) tomando a média do y
    tmp = tmp.groupby(x_col, as_index=False)[y_col].mean()
    tmp = tmp.sort_values(x_col).reset_index(drop=True)
    return tmp

def plot_continuo_plotly_from_bins(
    res_bins: pd.DataFrame,
    carteira: str,
    x_col: str,
    y_col: str,
    low_col: str,
    high_col: str,
    title: str,
    yaxis_title: str,
    ic_label: str,
    cdi_esperado_pct: float | None = None,
):
    sub = res_bins[res_bins["Carteira"] == carteira].copy()
    sub = sub.sort_values("_range_order").dropna(subset=[x_col, y_col, low_col, high_col])

    # garante numérico e remove duplicata no x também para low/high (fazendo média)
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub[low_col] = pd.to_numeric(sub[low_col], errors="coerce")
    sub[high_col] = pd.to_numeric(sub[high_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col, low_col, high_col])

    if sub.empty:
        st.warning(f"{carteira}: sem pontos válidos após filtros.")
        return

    # dedup por x (média)
    sub = sub.groupby(x_col, as_index=False).agg({
        y_col: "mean",
        low_col: "mean",
        high_col: "mean",
        "CDI Range (% a.a.)": "first",
        "Observações": "sum",
        "_range_order": "min"
    }).sort_values(x_col).reset_index(drop=True)

    if sub[x_col].nunique() < 2:
        st.warning(f"{carteira}: precisa de >=2 pontos (x únicos) para interpolar.")
        return

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    y_low = sub[low_col].to_numpy(dtype=float)
    y_high = sub[high_col].to_numpy(dtype=float)

    x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 300)

    mean_spline = PchipInterpolator(x, y)
    low_spline = PchipInterpolator(x, y_low)
    high_spline = PchipInterpolator(x, y_high)

    y_grid = mean_spline(x_grid)
    y_low_grid = low_spline(x_grid)
    y_high_grid = high_spline(x_grid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_grid, y=y_low_grid,
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_high_grid,
        mode="lines",
        fill="tonexty",
        name=f"Banda ({ic_label})",
        hoverinfo="skip"
    ))

    custom_line = np.stack([y_low_grid, y_high_grid], axis=1)
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode="lines",
        name="Linha (contínua)",
        line=dict(color=CARTEIRA_COLOR.get(carteira, "#333333"), width=2),
        customdata=custom_line,
        hovertemplate=(
            f"{carteira}<br>"
            "CDI midpoint (% a.a.): %{{x:.2f}}%<br>"
            "Média: %{{y:.2f}}<br>"
            f"Min (Lower {ic_label}): " + "%{customdata[0]:.2f}<br>"
            f"Max (Upper {ic_label}): " + "%{customdata[1]:.2f}"
            "<extra></extra>"
        )
    ))

    custom_pts = np.stack([
        sub["CDI Range (% a.a.)"].astype(str).to_numpy(),
        sub["Observações"].to_numpy(),
        sub[low_col].to_numpy(),
        sub[high_col].to_numpy(),
    ], axis=1)

    fig.add_trace(go.Scatter(
        x=sub[x_col],
        y=sub[y_col],
        mode="markers",
        name="Pontos (bins)",
        marker=dict(color=CARTEIRA_COLOR.get(carteira, "#333333"), size=7),
        customdata=custom_pts,
        hovertemplate=(
            f"{carteira}<br>"
            "CDI Range: %{customdata[0]}<br>"
            "Obs: %{customdata[1]}<br>"
            "CDI midpoint (% a.a.): %{{x:.2f}}%<br>"
            "Média: %{{y:.2f}}<br>"
            f"Min (Lower {ic_label}): " + "%{customdata[2]:.2f}<br>"
            f"Max (Upper {ic_label}): " + "%{customdata[3]:.2f}"
            "<extra></extra>"
        )
    ))

    if cdi_esperado_pct is not None and np.isfinite(cdi_esperado_pct):
        fig.add_vline(
            x=float(cdi_esperado_pct),
            line_width=2,
            line_dash="dash",
            line_color="#444444",
            annotation_text=f"CDI esp.: {cdi_esperado_pct:.2f}%",
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="CDI midpoint (% a.a.)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=420,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ✅ AQUI está a correção que resolve seu erro:
def plot_all_three_lines_no_ic(
    res_bins: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    yaxis_title: str,
    cdi_esperado_pct: float | None = None
):
    fig = go.Figure()
    any_trace = False

    for carteira in carteira_order:
        sub = res_bins[res_bins["Carteira"] == carteira].copy()
        sub = sub.sort_values("_range_order")

        tmp = _prepare_xy_unique(sub, x_col, y_col)

        # precisa de >= 2 pontos únicos
        if tmp.shape[0] < 2:
            continue

        x = tmp[x_col].to_numpy(dtype=float)
        y = tmp[y_col].to_numpy(dtype=float)

        # grid e PCHIP
        x_grid = np.linspace(float(x.min()), float(x.max()), 300)

        try:
            y_grid = PchipInterpolator(x, y)(x_grid)
        except Exception as e:
            st.warning(f"Falha ao interpolar {carteira} (provável x duplicado ou inválido): {e}")
            continue

        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid,
            mode="lines",
            name=carteira,
            line=dict(color=CARTEIRA_COLOR.get(carteira, "#333333"), width=2),
            hovertemplate=(
                f"{carteira}<br>"
                "CDI midpoint: %{{x:.2f}}%<br>"
                "Retorno: %{{y:.2f}}"
                "<extra></extra>"
            )
        ))
        any_trace = True

    if not any_trace:
        st.warning("Sem pontos suficientes para montar o comparativo (precisa de >=2 bins válidos por carteira).")
        return

    if cdi_esperado_pct is not None and np.isfinite(cdi_esperado_pct):
        fig.add_vline(
            x=float(cdi_esperado_pct),
            line_width=2,
            line_dash="dash",
            line_color="#444444",
            annotation_text=f"CDI esp.: {cdi_esperado_pct:.2f}%",
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="CDI midpoint (% a.a.)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_all_three_lines_linear_endpoints(
    res_bins: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    yaxis_title: str,
    cdi_esperado_pct: float | None = None
):
    fig = go.Figure()
    any_trace = False

    for carteira in carteira_order:
        sub = res_bins[res_bins["Carteira"] == carteira].copy()
        sub = sub.sort_values("_range_order")

        tmp = _prepare_xy_unique(sub, x_col, y_col)
        if tmp.shape[0] < 2:
            continue

        x0 = float(tmp[x_col].iloc[0]); y0 = float(tmp[y_col].iloc[0])
        x1 = float(tmp[x_col].iloc[-1]); y1 = float(tmp[y_col].iloc[-1])

        if x1 == x0:
            continue

        x_grid = np.linspace(x0, x1, 300)
        y_grid = y0 + (y1 - y0) * (x_grid - x0) / (x1 - x0)

        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid,
            mode="lines",
            name=f"{carteira} (reta)",
            line=dict(color=CARTEIRA_COLOR.get(carteira, "#333333"), width=2),
            hovertemplate=(
                f"{carteira} (reta)<br>"
                "CDI midpoint: %{{x:.2f}}%<br>"
                "Retorno: %{{y:.2f}}"
                "<extra></extra>"
            )
        ))
        any_trace = True

    if not any_trace:
        st.warning("Sem pontos suficientes para montar o comparativo (reta).")
        return

    if cdi_esperado_pct is not None and np.isfinite(cdi_esperado_pct):
        fig.add_vline(
            x=float(cdi_esperado_pct),
            line_width=2,
            line_dash="dash",
            line_color="#444444",
            annotation_text=f"CDI esp.: {cdi_esperado_pct:.2f}%",
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="CDI midpoint (% a.a.)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=520,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# LOAD
# =========================
st.caption(f"Arquivo: {DATA_PATH}")
if not DATA_PATH.exists():
    st.error("❌ Não encontrei o arquivo Dados.xlsx na mesma pasta do app.py.")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)

df_raw = load_data(DATA_PATH)

st.subheader("Preview do Excel")
st.dataframe(df_raw.head(15), use_container_width=True)

missing = [c for c in assets if c not in df_raw.columns]
if missing:
    st.error("❌ Faltam colunas no Excel: " + ", ".join(missing))
    st.stop()

# =========================
# COMPUTE FULL
# =========================
df_full = compute_everything(df_raw, window=WINDOW_ANUAL)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Parâmetros")

min_date = df_full.index.min().date()
max_date = df_full.index.max().date()

st.sidebar.subheader("Período de análise")
start_date = st.sidebar.date_input(
    "Data inicial (corta a amostra a partir daqui)",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
start_ts = pd.Timestamp(start_date)

st.sidebar.subheader("Intervalo de confiança (banda)")
ic_level_label = st.sidebar.selectbox("Nível do IC", ["68%", "90%", "95%"], index=1)
ic_level = float(ic_level_label.replace("%", "")) / 100.0
z = float(norm.ppf(0.5 + ic_level / 2.0))
st.sidebar.caption(f"Z usado: {z:.3f}")

st.sidebar.subheader("Faixas do CDI anual (em %)")
bin_start_pct = st.sidebar.number_input("Início (%)", value=2.0, step=0.5, format="%.1f")
bin_end_pct   = st.sidebar.number_input("Fim (%)", value=16.0, step=0.5, format="%.1f")
bin_step_pct  = st.sidebar.number_input("Passo (%)", value=2.0, step=0.5, format="%.1f", min_value=0.5)

if bin_end_pct <= bin_start_pct + bin_step_pct:
    st.error("Bins inválidos: ajuste Início/Fim/Passo (%).")
    st.stop()

bins = np.arange(1 + bin_start_pct/100.0, 1 + bin_end_pct/100.0 + 1e-9, bin_step_pct/100.0)

st.sidebar.subheader("Regime de Selic (tendência)")
trend_choice = st.sidebar.selectbox("Filtrar períodos por Selic", ["Todos", "Subindo", "Caindo"], index=0)

st.sidebar.subheader("CDI esperado (para marcar no gráfico)")
cdi_esperado_pct = st.sidebar.number_input("CDI esperado (% a.a.)", value=10.0, step=0.1, format="%.2f")

st.sidebar.subheader("Visualização: retorno base vs suavizado")
vis_tipo = st.sidebar.radio("Tipo de retorno", ["Base", "Suavizado"], index=1, horizontal=True)
vis_w = 5
if vis_tipo == "Suavizado":
    vis_w = st.sidebar.radio("Janela (w)", [3, 5], index=1, horizontal=True)

st.sidebar.subheader("Visualização: métrica")
vis_metric = st.sidebar.selectbox("Ver retornos como…", ["% do CDI", "Retorno Esperado (% a.a.)"], index=0)

# =========================
# APLICA CORTE POR DATA
# =========================
df = df_full[df_full.index >= start_ts].copy()

df_regime = df.copy()
if trend_choice in ["Subindo", "Caindo"]:
    df_regime = df_regime[df_regime["SelicTrend"] == trend_choice]

# =========================
# 0) HISTÓRICO — BASE100 + CDI + SELIC
# =========================
st.divider()
st.subheader("Evolução histórica — Carteiras (Base 100) + CDI (Base 100) e Selic anualizada")
st.altair_chart(make_dual_axis_selic_base100_with_cdi(df), use_container_width=True)

# =========================
# 1) TABELA + BARRAS
# =========================
st.divider()
st.subheader(f"Tabela — Retorno esperado por faixa de CDI anual (Selic: {trend_choice})")

results_df_base = build_cdi_regime_table(df_regime, bins=bins, z=z)
if results_df_base.empty:
    st.warning("Não houve observações suficientes nas faixas escolhidas (ou faltam dados na janela anual).")
    st.stop()

res_vis = prepare_visual_table_consistent(results_df_base, vis_tipo=vis_tipo, vis_w=vis_w)
label_suffix = " | Base" if vis_tipo == "Base" else f" | Suavizado (w={vis_w})"

range_order = (
    results_df_base[["CDI Range (% a.a.)", "_range_order"]]
    .drop_duplicates()
    .sort_values("_range_order")["CDI Range (% a.a.)"]
    .tolist()
)

table_cols = [
    "CDI Range (% a.a.)",
    "Carteira",
    "CDI Midpoint (% a.a.)",
    "VIS Expected Return (% a.a.)",
    "VIS Lower Bound (% a.a.)",
    "VIS Upper Bound (% a.a.)",
    "VIS Expected Return (% do CDI)",
    "VIS Lower Bound (% do CDI)",
    "VIS Upper Bound (% do CDI)",
    "Observações",
]
results_df_table = res_vis[table_cols].copy()
for c in table_cols:
    if c not in ["CDI Range (% a.a.)", "Carteira"]:
        results_df_table[c] = pd.to_numeric(results_df_table[c], errors="coerce").round(2)

results_df_table["Carteira"] = pd.Categorical(results_df_table["Carteira"], categories=carteira_order, ordered=True)
results_df_table["CDI Range (% a.a.)"] = pd.Categorical(results_df_table["CDI Range (% a.a.)"], categories=range_order, ordered=True)
results_df_table = results_df_table.sort_values(["CDI Range (% a.a.)", "Carteira"])
st.dataframe(results_df_table, use_container_width=True)

if vis_metric == "% do CDI":
    y_col = "VIS Expected Return (% do CDI)"
    low_col = "VIS Lower Bound (% do CDI)"
    high_col = "VIS Upper Bound (% do CDI)"
    y_title = "Expected Return (% do CDI)"
    final_title = "Retorno (% do CDI)"
else:
    y_col = "VIS Expected Return (% a.a.)"
    low_col = "VIS Lower Bound (% a.a.)"
    high_col = "VIS Upper Bound (% a.a.)"
    y_title = "Expected Return (% a.a.)"
    final_title = "Retorno esperado (% a.a.)"

bar_base = res_vis[["CDI Range (% a.a.)", "_range_order", "Carteira", y_col]].copy()
bar_base = bar_base.rename(columns={
    "CDI Range (% a.a.)": "cdi_range",
    "_range_order": "range_order",
    "Carteira": "portfolio",
    y_col: "y_value",
})
bar_base["y_value"] = pd.to_numeric(bar_base["y_value"], errors="coerce")
bar_base["cdi_range"] = pd.Categorical(bar_base["cdi_range"], categories=range_order, ordered=True)
bar_base["portfolio"] = pd.Categorical(bar_base["portfolio"], categories=carteira_order, ordered=True)

st.subheader(f"Barras agrupadas — {vis_metric}{label_suffix} (IC {ic_level_label})")
bar_ok = bar_base.dropna(subset=["y_value"]).copy()
if bar_ok.empty:
    st.warning("Sem dados válidos para o gráfico de barras.")
else:
    st.altair_chart(make_grouped_bar_chart(bar_ok, y_field="y_value", y_title=y_title), use_container_width=True)

# =========================
# 2) VIEW CONTÍNUA (empilhada)
# =========================
st.divider()
st.subheader("View contínua (interativa) — Retorno vs CDI (a partir dos bins)")

for carteira in carteira_order:
    plot_continuo_plotly_from_bins(
        res_bins=res_vis,
        carteira=carteira,
        x_col="CDI Midpoint (% a.a.)",
        y_col=y_col,
        low_col=low_col,
        high_col=high_col,
        title=f"{carteira} — {vis_metric}{label_suffix} (IC {ic_level_label})",
        yaxis_title=y_title,
        ic_label=ic_level_label,
        cdi_esperado_pct=float(cdi_esperado_pct),
    )

# =========================
# 3) COMPARATIVO (PCHIP, sem IC)
# =========================
st.divider()
st.subheader("Comparativo — 3 carteiras na mesma curva (sem IC)")

plot_all_three_lines_no_ic(
    res_bins=res_vis,
    x_col="CDI Midpoint (% a.a.)",
    y_col=y_col,
    title=f"Conservador vs Moderado vs Agressivo — {final_title}{label_suffix}",
    yaxis_title=y_title,
    cdi_esperado_pct=float(cdi_esperado_pct),
)

# =========================
# 4) COMPARATIVO (reta extremos, sem IC)
# =========================
st.divider()
st.subheader("Comparativo — reta linear ligando o extremo esquerdo ao extremo direito (sem IC)")

plot_all_three_lines_linear_endpoints(
    res_bins=res_vis,
    x_col="CDI Midpoint (% a.a.)",
    y_col=y_col,
    title=f"Conservador vs Moderado vs Agressivo — reta entre extremos — {final_title}{label_suffix}",
    yaxis_title=y_title,
    cdi_esperado_pct=float(cdi_esperado_pct),
)

with st.expander("Diagnóstico (regime filtrado): colunas anuais usadas"):
    cols_diag = ["CDIAcumuladoAnual", "SelicAnualizada", "SelicTrend"] + [f"FatorAnual_{n}" for n in carteira_order]
    st.dataframe(df_regime[cols_diag].dropna().head(50), use_container_width=True)
