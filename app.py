import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# =========================
# THEME / CORES
# =========================
PRIMARY_COLORS = ["#08082a", "#dbdbdb", "#dfac16"]  # Conservador, Moderado, Agressivo
SECONDARY_COLORS = ["#f7e4af", "#999999", "#c00000", "#dce6f2"]  # muitas linhas
SELIC_COLOR = "#c00000"  # Selic anualizada (linha)

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

# Janela anual fixa (dias úteis)
WINDOW_ANUAL = 252

# =========================
# FUNÇÕES
# =========================
def rolling_annual_factor(series: pd.Series, window: int) -> pd.Series:
    """Fator anual rolling: exp(sum(log(1+r))) em janela 'window'. Ex: 1.13 = +13%."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.exp(np.log1p(s).rolling(window=window).sum())

def calculate_metrics(factor_series: pd.Series, z: float = 1.645):
    """Média e IC (média ± z*std) no espaço de FATOR (ex: 1.10)."""
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

    # 1ª coluna = data
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # numéricos nas classes (retornos diários)
    df[assets] = df[assets].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Selic anualizada a partir do CDI diário:
    # r_aa = (1 + r_dia) ^ 252 - 1
    df["SelicAnualizada"] = (1.0 + df["CDI"]).pow(window) - 1.0  # em decimal (ex: 0.135)

    # tendência da selic (subindo/caindo) pelo delta diário do anualizado
    df["SelicTrend"] = np.where(df["SelicAnualizada"].diff() >= 0, "Subindo", "Caindo")

    # Base 100 das classes
    for col in assets:
        df[f"Base100_{col}"] = 100 * (1 + df[col]).cumprod()

    # Carteiras: retorno diário, base 100 e anual rolling (fator)
    for name, wdict in portfolios.items():
        w = np.array([wdict.get(a, 0.0) for a in assets], dtype=float)
        df[f"Retorno_{name}"] = df[assets].values @ w
        df[f"Base100_{name}"] = 100 * (1 + df[f"Retorno_{name}"]).cumprod()
        df[f"FatorAnual_{name}"] = rolling_annual_factor(df[f"Retorno_{name}"], window=window)

    # CDI anual rolling (fator) para bins
    df["CDIAcumuladoAnual"] = rolling_annual_factor(df["CDI"], window=window)

    return df

def build_cdi_regime_table(df: pd.DataFrame, bins: np.ndarray, z: float = 1.645) -> pd.DataFrame:
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
                "Carteira": name,
                "Expected Return (% a.a.)": exp_pct,
                "Lower Bound 90% (% a.a.)": lower_pct,
                "Upper Bound 90% (% a.a.)": upper_pct,
                "Expected Return (% do CDI)": exp_pct_cdi,
                "Lower Bound (% do CDI)": lower_pct_cdi,
                "Upper Bound (% do CDI)": upper_pct_cdi,
                "Observações": nobs,
            })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    for c in ["Expected Return (% a.a.)", "Lower Bound 90% (% a.a.)", "Upper Bound 90% (% a.a.)",
              "Expected Return (% do CDI)", "Lower Bound (% do CDI)", "Upper Bound (% do CDI)"]:
        res[c] = pd.to_numeric(res[c], errors="coerce").round(2)

    res["Carteira"] = pd.Categorical(res["Carteira"], categories=carteira_order, ordered=True)
    res = res.sort_values(["_range_order", "Carteira"])
    return res

def make_grouped_bar_chart(bar_long: pd.DataFrame, y_field: str, y_title: str) -> alt.Chart:
    color_scale = alt.Scale(domain=carteira_order, range=PRIMARY_COLORS)

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

def make_base100_line_chart(df: pd.DataFrame, selected_assets: list[str]) -> alt.Chart:
    cols = [f"Base100_{c}" for c in selected_assets]
    tmp = df[cols].copy()
    tmp.columns = selected_assets

    long_df = tmp.reset_index().melt(
        id_vars=[tmp.reset_index().columns[0]],
        var_name="classe",
        value_name="base100"
    )
    date_col = long_df.columns[0]
    long_df = long_df.rename(columns={date_col: "data"})

    palette = SECONDARY_COLORS + PRIMARY_COLORS
    colors = (palette * (len(selected_assets) // len(palette) + 1))[:len(selected_assets)]
    line_scale = alt.Scale(domain=selected_assets, range=colors)

    return (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("data:T", title="Data"),
            y=alt.Y("base100:Q", title="Base 100"),
            color=alt.Color("classe:N", scale=line_scale, legend=alt.Legend(title="Classe")),
            tooltip=[
                alt.Tooltip("data:T", title="Data"),
                alt.Tooltip("classe:N", title="Classe"),
                alt.Tooltip("base100:Q", title="Base 100", format=".2f"),
            ],
        )
        .properties(height=520)
    )

def make_dual_axis_selic_base100(df: pd.DataFrame) -> alt.Chart:
    """
    Eixo esquerdo: Base100 das 3 carteiras
    Eixo direito: SelicAnualizada (% a.a.)
    """
    # Base100 portfolios em formato long
    base_cols = [f"Base100_{n}" for n in carteira_order]
    tmp = df[base_cols].copy()
    tmp.columns = carteira_order
    long_base = tmp.reset_index().melt(id_vars=[tmp.reset_index().columns[0]], var_name="portfolio", value_name="base100")
    date_col = long_base.columns[0]
    long_base = long_base.rename(columns={date_col: "data"})

    # Selic anualizada (%)
    selic_df = df[["SelicAnualizada"]].copy()
    selic_df = selic_df.reset_index().rename(columns={selic_df.reset_index().columns[0]: "data"})
    selic_df["selic_aa_pct"] = selic_df["SelicAnualizada"] * 100.0

    # Escalas de cor
    port_color = alt.Scale(domain=carteira_order, range=PRIMARY_COLORS)

    base_lines = (
        alt.Chart(long_base)
        .mark_line()
        .encode(
            x=alt.X("data:T", title="Data"),
            y=alt.Y("base100:Q", title="Base 100"),
            color=alt.Color("portfolio:N", scale=port_color, legend=alt.Legend(title="Carteira")),
            tooltip=[
                alt.Tooltip("data:T", title="Data"),
                alt.Tooltip("portfolio:N", title="Carteira"),
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
        .properties(height=520, title="Selic anualizada (eixo direito) vs Base 100 das carteiras (eixo esquerdo)")
    )

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
# SIDEBAR
# =========================
st.sidebar.header("Parâmetros")

z = st.sidebar.number_input("Z (IC ~90%: 1.645)", value=1.645, step=0.001, format="%.3f")

st.sidebar.subheader("Faixas do CDI anual (em %)")
bin_start_pct = st.sidebar.number_input("Início (%)", value=2.0, step=1.0, format="%.1f")
bin_end_pct   = st.sidebar.number_input("Fim (%)", value=15.0, step=1.0, format="%.1f")
bin_step_pct  = st.sidebar.number_input("Passo (%)", value=3.0, step=1.0, format="%.1f")

if bin_end_pct <= bin_start_pct + bin_step_pct:
    st.error("Bins inválidos: ajuste Início/Fim/Passo (%).")
    st.stop()

bins = np.arange(1 + bin_start_pct/100.0, 1 + bin_end_pct/100.0 + 1e-9, bin_step_pct/100.0)

st.sidebar.subheader("Regime de Selic (tendência)")
trend_choice = st.sidebar.selectbox("Filtrar períodos por Selic", ["Todos", "Subindo", "Caindo"], index=0)

# =========================
# COMPUTE
# =========================
df = compute_everything(df_raw, window=WINDOW_ANUAL)

# aplica filtro de tendência (para tabela/bins/barras)
df_regime = df.copy()
if trend_choice in ["Subindo", "Caindo"]:
    df_regime = df_regime[df_regime["SelicTrend"] == trend_choice]

# =========================
# 0) NOVO GRÁFICO: EIXO DUPLO (Selic anualizada + Base100 carteiras)
# =========================
st.divider()
st.subheader("Selic anualizada vs Base 100 das carteiras (eixo duplo)")
st.altair_chart(make_dual_axis_selic_base100(df), use_container_width=True)

# =========================
# 1) BASE 100 — LINHAS (classes)
# =========================
st.divider()
st.subheader("Evolução das classes (Base 100)")

selected_assets = st.multiselect(
    "Escolha as classes para visualizar",
    options=assets,
    default=assets
)

if selected_assets:
    st.altair_chart(make_base100_line_chart(df, selected_assets), use_container_width=True)
else:
    st.info("Selecione ao menos 1 classe para plotar.")

# =========================
# 2) TABELA + GRÁFICOS DE BARRAS
# =========================
st.divider()
st.subheader(f"Tabela — Retorno esperado por faixa de CDI anual  (Selic: {trend_choice})")

results_df = build_cdi_regime_table(df_regime, bins=bins, z=z)

if results_df.empty:
    st.warning("Não houve observações suficientes nas faixas escolhidas (ou faltam dados na janela anual).")
    st.stop()

# ordem correta dos ranges (menor -> maior)
range_order = (
    results_df[["CDI Range (% a.a.)", "_range_order"]]
    .drop_duplicates()
    .sort_values("_range_order")["CDI Range (% a.a.)"]
    .tolist()
)

# tabela ordenada: range e carteira
results_df_table = results_df.drop(columns=["_range_order"]).copy()
results_df_table["Carteira"] = pd.Categorical(results_df_table["Carteira"], categories=carteira_order, ordered=True)
results_df_table["CDI Range (% a.a.)"] = pd.Categorical(
    results_df_table["CDI Range (% a.a.)"], categories=range_order, ordered=True
)
results_df_table = results_df_table.sort_values(["CDI Range (% a.a.)", "Carteira"])
st.dataframe(results_df_table, use_container_width=True)

# dataset para barras (com ordem numérica)
bar_base = results_df[["CDI Range (% a.a.)", "_range_order", "Carteira",
                       "Expected Return (% a.a.)", "Expected Return (% do CDI)"]].copy()

bar_base = bar_base.rename(columns={
    "CDI Range (% a.a.)": "cdi_range",
    "_range_order": "range_order",
    "Carteira": "portfolio",
    "Expected Return (% a.a.)": "expected_return_aa",
    "Expected Return (% do CDI)": "expected_return_cdi",
})

for col in ["expected_return_aa", "expected_return_cdi"]:
    bar_base[col] = pd.to_numeric(bar_base[col], errors="coerce")

# força ordem do eixo X e da carteira
bar_base["cdi_range"] = pd.Categorical(bar_base["cdi_range"], categories=range_order, ordered=True)
bar_base["portfolio"] = pd.Categorical(bar_base["portfolio"], categories=carteira_order, ordered=True)

# GRÁFICO 1: % a.a.
st.subheader("Barras agrupadas — Expected Return (% a.a.) por faixa de CDI")
bar_aa = bar_base.dropna(subset=["expected_return_aa"]).copy()
if bar_aa.empty:
    st.warning("Sem dados válidos para o gráfico de % a.a. (verifique bins/janela).")
else:
    st.altair_chart(
        make_grouped_bar_chart(bar_aa, y_field="expected_return_aa", y_title="Expected Return (% a.a.)"),
        use_container_width=True
    )

# GRÁFICO 2: % do CDI
st.subheader("Barras agrupadas — Expected Return (% do CDI) por faixa de CDI")
bar_cdi = bar_base.dropna(subset=["expected_return_cdi"]).copy()
if bar_cdi.empty:
    st.warning("Sem dados válidos para o gráfico de % do CDI (verifique bins/janela).")
else:
    st.altair_chart(
        make_grouped_bar_chart(bar_cdi, y_field="expected_return_cdi", y_title="Expected Return (% do CDI)"),
        use_container_width=True
    )

with st.expander("Diagnóstico (regime filtrado): colunas anuais usadas"):
    cols_diag = ["CDIAcumuladoAnual", "SelicAnualizada", "SelicTrend"] + [f"FatorAnual_{n}" for n in carteira_order]
    st.dataframe(df_regime[cols_diag].dropna().head(30), use_container_width=True)
