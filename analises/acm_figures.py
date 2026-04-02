"""
Regenera todas as figuras do survey com barras mais finas,
em versões PT e EN. Mesmo estilo visual dos gráficos originais.

Usage:
    python3 analises/acm_figures.py

Output:
    output/acm/pt/  — Portuguese
    output/acm/en/  — English
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import seaborn as sns

# ── Constantes ──────────────────────────────────────────────────────────────
BASE_ORANGE = "#ff6002"
BOX_TARGET_WIDTH_IN = 0.14  # target visual width per box in inches
DPI = 300
FIG_EXT = ".png"

# ACM single-column dimensions (inches)
# Column width ~3.33in. Heights adjusted per number of categories.
FIG_W = 3.33
FIG_H_SMALL = 2.4   # ≤5 groups (age quartiles, regions)
FIG_H_MEDIUM = 2.8  # 6 groups (experience, levels)
FIG_H_LARGE = 3.5   # 8+ groups (areas)
FIG_H_XLARGE = 4.2  # 14+ groups (states)
FIG_H_SCATTER = 2.4  # scatter plot

# Font sizes scaled for column width
FONT_LABEL = 8
FONT_TICK = 7
FONT_MEAN = 6
FONT_STAT = 6
FONT_LEGEND = 6
MEAN_S = 30          # marker size
MEAN_LW = 0.8        # marker edge
LINE_WIDTH = 1.0      # box lines
STROKE_WIDTH = 1.5    # text stroke

LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

REGION_MAP = {
    "Acre": "Norte", "Amapá": "Norte", "Amazonas": "Norte", "Pará": "Norte",
    "Rondônia": "Norte", "Roraima": "Norte", "Tocantins": "Norte",
    "Alagoas": "Nordeste", "Bahia": "Nordeste", "Ceará": "Nordeste",
    "Maranhão": "Nordeste", "Paraíba": "Nordeste", "Pernambuco": "Nordeste",
    "Piauí": "Nordeste", "Rio Grande do Norte": "Nordeste", "Sergipe": "Nordeste",
    "Distrito Federal": "Centro-Oeste", "Goiás": "Centro-Oeste",
    "Mato Grosso": "Centro-Oeste", "Mato Grosso do Sul": "Centro-Oeste",
    "Espírito Santo": "Sudeste", "Minas Gerais": "Sudeste",
    "Rio de Janeiro": "Sudeste", "São Paulo": "Sudeste",
    "Paraná": "Sul", "Rio Grande do Sul": "Sul", "Santa Catarina": "Sul",
}

STATE_ABBR = {
    "Acre": "AC", "Alagoas": "AL", "Amapá": "AP", "Amazonas": "AM",
    "Bahia": "BA", "Ceará": "CE", "Distrito Federal": "DF",
    "Espírito Santo": "ES", "Goiás": "GO", "Maranhão": "MA",
    "Mato Grosso": "MT", "Mato Grosso do Sul": "MS", "Minas Gerais": "MG",
    "Pará": "PA", "Paraíba": "PB", "Paraná": "PR", "Pernambuco": "PE",
    "Piauí": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "Rondônia": "RO", "Roraima": "RR",
    "Santa Catarina": "SC", "São Paulo": "SP", "Sergipe": "SE",
    "Tocantins": "TO",
}

EXPERIENCIA_ORDER = [
    "Até um ano.", "Entre 1 e 2 anos.", "Entre 3 e 4 anos.",
    "Entre 5 e 6 anos.", "Entre 7 e 8 anos.", "Mais de 8 anos.",
]
EXP_ABBR = {"Até um ano.": "<1", "Entre 1 e 2 anos.": "1-2",
            "Entre 3 e 4 anos.": "3-4", "Entre 5 e 6 anos.": "5-6",
            "Entre 7 e 8 anos.": "7-8", "Mais de 8 anos.": ">8"}

NIVEL_ORDER = [
    "Em treinamento", "Júnior", "Pleno", "Sênior",
    "Especialista (Foco técnico)", "Gerente (Foco em pessoas)",
]
NIVEL_LABELS_PT = {
    "Em treinamento": "Em treinamento", "Júnior": "Júnior", "Pleno": "Pleno",
    "Sênior": "Sênior", "Especialista (Foco técnico)": "Especialista\n(Foco técnico)",
    "Gerente (Foco em pessoas)": "Gerente\n(Foco em pessoas)",
}
NIVEL_LABELS_EN = {
    "Em treinamento": "Trainee", "Júnior": "Junior", "Pleno": "Mid-level",
    "Sênior": "Senior", "Especialista (Foco técnico)": "Specialist\n(Technical)",
    "Gerente (Foco em pessoas)": "Manager\n(People)",
}

REGION_EN = {
    "Norte": "North", "Nordeste": "Northeast", "Centro-Oeste": "Midwest",
    "Sudeste": "Southeast", "Sul": "South",
}

AREA_LABELS_PT = {
    "Arquitetura": "Arquitetura",
    "Área de dados": "Área de dados",
    "Controle de Qualidade (QA)": "Controle de\nQualidade (QA)",
    "Design (UI/UX)": "Design\n(UI/UX)",
    "Desenvolvimento": "Desenvolvimento",
    "DevOps": "DevOps",
    "Gestão de Pessoas / Projetos": "Gestão de\nPessoas /\nProjetos",
    "Liderança Técnica": "Liderança\nTécnica",
}
AREA_LABELS_EN = {
    "Arquitetura": "Architecture",
    "Área de dados": "Data",
    "Controle de Qualidade (QA)": "QA",
    "Design (UI/UX)": "Design\n(UI/UX)",
    "Desenvolvimento": "Development",
    "DevOps": "DevOps",
    "Gestão de Pessoas / Projetos": "People /\nProject Mgmt",
    "Liderança Técnica": "Tech Lead",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _orange_shades(n: int) -> list:
    base_rgb = np.array(mcolors.to_rgb(BASE_ORANGE))
    white = np.array([1.0, 1.0, 1.0])
    if n <= 1:
        return [tuple(base_rgb.tolist())]
    shades = []
    for i in range(n):
        t = i / max(1, n - 1)
        mix = 0.35 + 0.65 * t
        rgb = white * (1.0 - mix) + base_rgb * mix
        shades.append(tuple(rgb.tolist()))
    return shades


def find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for col in df.columns:
        if keyword.lower() in str(col).lower():
            return col
    return None


def clean_estado(s) -> Optional[str]:
    if pd.isna(s):
        return None
    return re.sub(r"\s*\([A-Z]{2}\)\s*$", "", str(s).strip()) or None


def convert_age(s) -> Optional[float]:
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    age_map = {
        "até 20 anos": 20, "21 a 30 anos": 25.5, "31 a 40 anos": 35.5,
        "41 a 50 anos": 45.5, "51 a 60 anos": 55.5, "mais de 60 anos": 65,
        "de 18 a 25 anos": 21.5, "de 26 a 35 anos": 30.5,
        "de 36 a 45 anos": 40.5, "de 46 a 55 anos": 50.5,
        "acima de 55 anos": 60,
    }
    for k, v in age_map.items():
        if k in s:
            return float(v)
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else None


def sig_text(p: float, lang: str = "pt") -> Tuple[str, str]:
    if lang == "en":
        if p < 0.001:
            return "***", "Highly significant"
        elif p < 0.01:
            return "**", "Very significant"
        elif p < 0.05:
            return "*", "Significant"
        else:
            return "ns", "Not significant"
    else:
        if p < 0.001:
            return "***", "Altamente significativo"
        elif p < 0.01:
            return "**", "Muito significativo"
        elif p < 0.05:
            return "*", "Significativo"
        else:
            return "ns", "Não significativo"


def stat_box_text(test_name: str, n: int, stat_val: float, stat_label: str,
                  p_val: float, lang: str) -> str:
    sym, txt = sig_text(p_val, lang)
    result_label = "Result" if lang == "en" else "Resultado"
    return (
        f"{test_name}\n"
        f"{'─' * 32}\n"
        f"n = {n}\n"
        f"{stat_label} = {stat_val:.4f}\n"
        f"p = {p_val:.6f}\n"
        f"{result_label}: {sym} {txt}"
    )


# ── Generic boxplot (same style as originals, thinner bars) ─────────────────
def plot_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: List[str],
    x_labels: List[str],
    xlabel: str,
    ylabel: str,
    stat_box: Optional[str],
    stat_box_pos: Tuple[float, float, str, str],
    out_path: Path,
    figsize: Tuple[float, float] = (FIG_W, FIG_H_MEDIUM),
    x_rotation: float = 0,
    x_ha: str = "center",
    legend_loc: str = "upper right",
    mean_label: str = "Média",
):
    palette = _orange_shades(len(order))
    color_dict = {g: palette[i] for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=figsize)

    # Compute box width so all charts have the same visual width per box
    n_groups = len(order)
    box_width = min(0.6, BOX_TARGET_WIDTH_IN * n_groups / figsize[0])

    sns.boxplot(
        data=df, x=x_col, y=y_col, order=order,
        hue=x_col, palette=color_dict, legend=False,
        ax=ax, width=box_width, linewidth=LINE_WIDTH,
        showfliers=False,
    )

    # Médias como bolinhas pretas
    means = df.groupby(x_col, observed=True)[y_col].mean()
    positions = range(len(order))
    means_ordered = [means.get(g, np.nan) for g in order]

    ax.scatter(
        positions, means_ordered, color="black", s=MEAN_S, zorder=3,
        marker="o", edgecolor="white", linewidth=MEAN_LW, label=mean_label,
    )

    # Valores das médias
    for pos, g in enumerate(order):
        m = means_ordered[pos]
        if np.isnan(m):
            continue
        text = ax.text(
            pos, m + 0.15, f"{m:.2f}",
            ha="center", va="bottom", fontsize=FONT_MEAN,
            fontweight="bold", color="black",
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=STROKE_WIDTH, foreground="white"),
            path_effects.Normal(),
        ])

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(x_labels, fontsize=FONT_TICK, rotation=x_rotation, ha=x_ha)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.4)
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc=legend_loc, fontsize=FONT_LEGEND)

    # Stat box
    if stat_box:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8,
                     edgecolor="black", linewidth=0.6)
        ax.text(
            stat_box_pos[0], stat_box_pos[1], stat_box,
            transform=ax.transAxes, fontsize=FONT_STAT,
            verticalalignment=stat_box_pos[2],
            horizontalalignment=stat_box_pos[3],
            fontfamily="monospace", bbox=props,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    xlabel: str, ylabel: str,
    stat_box: str,
    out_path: Path,
    figsize: Tuple[float, float] = (FIG_W, FIG_H_SCATTER),
):
    fig, ax = plt.subplots(figsize=figsize)

    sns.regplot(
        data=df, x=x_col, y=y_col,
        scatter_kws=dict(s=12, alpha=0.6, color=BASE_ORANGE,
                         edgecolor="white", linewidths=0.3),
        line_kws=dict(color=BASE_ORANGE, linewidth=1.2),
        ax=ax,
    )

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8,
                 edgecolor="black", linewidth=0.6)
    ax.text(
        0.05, 0.95, stat_box,
        transform=ax.transAxes, fontsize=FONT_STAT, fontweight="bold",
        verticalalignment="top", fontfamily="monospace", bbox=props,
        color="black",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Kruskal-Wallis helper ───────────────────────────────────────────────────
def kruskal_groups(df: pd.DataFrame, group_col: str, value_col: str,
                   min_n: int = 5) -> Tuple[Optional[float], Optional[float], int]:
    grupos = {}
    for g in df[group_col].unique():
        vals = df[df[group_col] == g][value_col].dropna().values
        if len(vals) >= min_n:
            grupos[g] = vals
    if len(grupos) < 2:
        return None, None, 0
    h, p = stats.kruskal(*grupos.values())
    n_total = sum(len(v) for v in grupos.values())
    return h, p, n_total


# ── Data preparation ────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, engine="python")


def prep_experience(df: pd.DataFrame, q_keyword: str):
    col_exp = find_col(df, "tempo total de experiência")
    col_q = find_col(df, q_keyword)
    d = df[[col_exp, col_q]].copy()
    d.columns = ["group", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d = d[d["group"].isin(EXPERIENCIA_ORDER)].dropna()
    d["group"] = pd.Categorical(d["group"], categories=EXPERIENCIA_ORDER, ordered=True)
    return d


def prep_region(df: pd.DataFrame, q_keyword: str):
    estado_col = find_col(df, "estado")
    q_col = find_col(df, q_keyword)
    d = df[[estado_col, q_col]].copy()
    d.columns = ["estado", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d["estado"] = d["estado"].apply(clean_estado)
    d["group"] = d["estado"].map(REGION_MAP)
    d = d.dropna(subset=["group", "value"])
    return d[["group", "value"]].copy()


def prep_state(df: pd.DataFrame, q_keyword: str, min_n: int = 5):
    estado_col = find_col(df, "estado")
    q_col = find_col(df, q_keyword)
    d = df[[estado_col, q_col]].copy()
    d.columns = ["estado_full", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d["estado_full"] = d["estado_full"].apply(clean_estado)
    d["group"] = d["estado_full"].map(STATE_ABBR)
    d = d.dropna(subset=["group", "value"])
    counts = d["group"].value_counts()
    valid = counts[counts >= min_n].index.tolist()
    d = d[d["group"].isin(valid)]
    means = d.groupby("group")["value"].mean().sort_values()
    order = means.index.tolist()
    return d[["group", "value"]].copy(), order


def prep_area(df: pd.DataFrame, q_keyword: str, min_n: int = 5):
    area_col = find_col(df, "principal área de atuação")
    q_col = find_col(df, q_keyword)
    d = df[[area_col, q_col]].copy()
    d.columns = ["group", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d = d.dropna()
    counts = d["group"].value_counts()
    valid = counts[counts >= min_n].index.tolist()
    d = d[d["group"].isin(valid)]
    means = d.groupby("group")["value"].mean().sort_values()
    order = means.index.tolist()
    return d, order


def prep_nivel(df: pd.DataFrame, q_keyword: str):
    col_nivel = find_col(df, "nível profissional")
    col_q = find_col(df, q_keyword)
    d = df[[col_nivel, col_q]].copy()
    d.columns = ["group", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d = d[d["group"].isin(NIVEL_ORDER)].dropna()
    d["group"] = pd.Categorical(d["group"], categories=NIVEL_ORDER, ordered=True)
    return d


def prep_age_groups(df: pd.DataFrame, q_keyword: str):
    age_col = find_col(df, "qual é a sua idade")
    if age_col is None:
        age_col = find_col(df, "idade")
    q_col = find_col(df, q_keyword)
    d = df[[age_col, q_col]].copy()
    d.columns = ["age", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d["age"] = d["age"].apply(convert_age)
    d = d.dropna()
    quartiles = pd.qcut(d["age"], q=4, labels=False, duplicates="drop")
    d["group"] = quartiles.apply(lambda q: "Q1" if q == 0 else ("Q2+Q3" if q in (1, 2) else "Q4"))
    ranges = {}
    for g in ["Q1", "Q2+Q3", "Q4"]:
        mask = d["group"] == g
        if mask.any():
            ranges[g] = (int(d.loc[mask, "age"].min()), int(d.loc[mask, "age"].max()))
    d["group"] = pd.Categorical(d["group"], categories=["Q1", "Q2+Q3", "Q4"], ordered=True)
    return d, ranges


def prep_age_scatter(df: pd.DataFrame, q_keyword: str):
    age_col = find_col(df, "qual é a sua idade")
    if age_col is None:
        age_col = find_col(df, "idade")
    q_col = find_col(df, q_keyword)
    d = df[[age_col, q_col]].copy()
    d.columns = ["age", "value"]
    d["value"] = d["value"].map(LIKERT_MAP).astype(float)
    d["age"] = d["age"].apply(convert_age)
    return d.dropna()


# ── Generate all figures ────────────────────────────────────────────────────
def generate_all(csv_path: Path, base_out: Path):
    df = load_csv(csv_path)
    print(f"Loaded {len(df)} rows\n")

    for lang in ["pt", "en"]:
        out_dir = base_out / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        is_en = lang == "en"
        mean_lbl = "Mean" if is_en else "Média"
        print(f"=== {lang.upper()} ===")

        # ────────────────────────────────────────────────────────────────
        # 1. Q6 age groups (boxplot)
        # ────────────────────────────────────────────────────────────────
        data_ag, ranges = prep_age_groups(df, "carga horária")
        order_ag = ["Q1", "Q2+Q3", "Q4"]
        labels_ag = []
        for g in order_ag:
            r = ranges.get(g, (0, 0))
            n = (data_ag["group"] == g).sum()
            labels_ag.append(f"{g} ({r[0]}-{r[1]})\n(n={n})")

        h, p, n_t = kruskal_groups(data_ag, "group", "value")
        test_name = "Kruskal-Wallis Test" if is_en else "Teste de Kruskal-Wallis"
        sbox = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_ag = "Age Groups" if is_en else "Grupos de Idade"
        ylabel_ag = "Q6 (Personal time impact)" if is_en else "Q6 (Impacto no tempo pessoal)"
        fname_ag = "Q6_age_groups.png" if is_en else "Q6_tempo_pessoal_agrupado.png"

        plot_boxplot(
            data_ag, "group", "value", order_ag, labels_ag,
            xlabel_ag, ylabel_ag, sbox, (0.02, 0.95, "top", "left"),
            out_dir / fname_ag, figsize=(FIG_W, FIG_H_SMALL), mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 2. Q6 scatter (age vs Q6)
        # ────────────────────────────────────────────────────────────────
        data_sc = prep_age_scatter(df, "carga horária")
        r_val, p_val = stats.spearmanr(data_sc["age"], data_sc["value"])
        sym, txt = sig_text(p_val, lang)
        corr_name = "Spearman Correlation" if is_en else "Correlação de Spearman"
        result_lbl = "Result" if is_en else "Resultado"
        sbox_sc = (
            f"{corr_name}\n"
            f"{'─' * 32}\n"
            f"n = {len(data_sc)}\n"
            f"r = {r_val:.4f}\n"
            f"p = {p_val:.6f}\n"
            f"{result_lbl}: {sym} {txt}"
        )
        xlabel_sc = "Age (years)" if is_en else "Idade (anos)"
        ylabel_sc = "Q6 (Personal time impact)" if is_en else "Q6 (Impacto no tempo pessoal)"
        fname_sc = "Q6_age_scatter.png" if is_en else "Q6_tempo_pessoal.png"

        plot_scatter(
            data_sc, "age", "value", xlabel_sc, ylabel_sc,
            sbox_sc, out_dir / fname_sc,
        )

        # ────────────────────────────────────────────────────────────────
        # 3. Q7 by state
        # ────────────────────────────────────────────────────────────────
        data_st, order_st = prep_state(df, "competitivo")
        labels_st = []
        for g in order_st:
            n = (data_st["group"] == g).sum()
            labels_st.append(f"{g}\n(n={n})")

        h, p, n_t = kruskal_groups(data_st, "group", "value")
        sbox_st = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_st = "State" if is_en else "Estado"
        ylabel_st = "Q7 (Market competitiveness)" if is_en else "Q7 (Competitividade no mercado)"
        fname_st = "Q7_state.png" if is_en else "Q7_competitividade_estado.png"

        plot_boxplot(
            data_st, "group", "value", order_st, labels_st,
            xlabel_st, ylabel_st, sbox_st, (0.98, 0.02, "bottom", "right"),
            out_dir / fname_st, figsize=(FIG_W, FIG_H_XLARGE),
            x_rotation=90, x_ha="center", mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 4. Q7 by region
        # ────────────────────────────────────────────────────────────────
        data_rg = prep_region(df, "competitivo")
        means_rg = data_rg.groupby("group")["value"].mean().sort_values()
        order_rg = means_rg.index.tolist()
        labels_rg = []
        for g in order_rg:
            n = (data_rg["group"] == g).sum()
            lbl = REGION_EN[g] if is_en else g
            labels_rg.append(f"{lbl}\n(n={n})")

        h, p, n_t = kruskal_groups(data_rg, "group", "value")
        sbox_rg = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_rg = "Region" if is_en else "Região do Brasil"
        ylabel_rg = "Q7 (Market competitiveness)" if is_en else "Q7 (Competitividade no mercado)"
        fname_rg = "Q7_region.png" if is_en else "Q7_competitividade_regiao.png"

        plot_boxplot(
            data_rg, "group", "value", order_rg, labels_rg,
            xlabel_rg, ylabel_rg, sbox_rg, (0.98, 0.02, "bottom", "right"),
            out_dir / fname_rg, figsize=(FIG_W, FIG_H_SMALL), mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 5. Q8 by area
        # ────────────────────────────────────────────────────────────────
        data_a8, order_a8 = prep_area(df, "motivou a buscar")
        area_lbl_map = AREA_LABELS_EN if is_en else AREA_LABELS_PT
        labels_a8 = []
        for g in order_a8:
            n = (data_a8["group"] == g).sum()
            lbl = area_lbl_map.get(g, g)
            labels_a8.append(f"{lbl}\n(n={n})")

        h, p, n_t = kruskal_groups(data_a8, "group", "value")
        sbox_a8 = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_a8 = "Professional Area (ordered by ascending mean)" if is_en else "Área de Atuação (ordenado por motivação crescente)"
        ylabel_a8 = "Q8 (Learning motivation)" if is_en else "Q8 (Motivação para aprender)"
        fname_a8 = "Q8_area.png" if is_en else "Q8_motivação_aprender.png"

        plot_boxplot(
            data_a8, "group", "value", order_a8, labels_a8,
            xlabel_a8, ylabel_a8, sbox_a8, (0.98, 0.05, "bottom", "right"),
            out_dir / fname_a8, figsize=(FIG_W, FIG_H_LARGE),
            x_rotation=90, x_ha="center", legend_loc="upper left",
            mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 6. Q10 by state
        # ────────────────────────────────────────────────────────────────
        data_s10, order_s10 = prep_state(df, "tempo suficiente durante o expediente")
        labels_s10 = []
        for g in order_s10:
            n = (data_s10["group"] == g).sum()
            labels_s10.append(f"{g}\n(n={n})")

        h, p, n_t = kruskal_groups(data_s10, "group", "value")
        sbox_s10 = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_s10 = "State" if is_en else "Estado"
        ylabel_s10 = "Q10 (Work hours sufficiency)" if is_en else "Q10 (Tempo no expediente)"
        fname_s10 = "Q10_state.png" if is_en else "Q10_tempo_expediente_estado.png"

        plot_boxplot(
            data_s10, "group", "value", order_s10, labels_s10,
            xlabel_s10, ylabel_s10, sbox_s10, (0.98, 0.02, "bottom", "right"),
            out_dir / fname_s10, figsize=(FIG_W, FIG_H_XLARGE),
            x_rotation=90, x_ha="center", mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 7. Q10 by region
        # ────────────────────────────────────────────────────────────────
        data_r10 = prep_region(df, "tempo suficiente durante o expediente")
        means_r10 = data_r10.groupby("group")["value"].mean().sort_values()
        order_r10 = means_r10.index.tolist()
        labels_r10 = []
        for g in order_r10:
            n = (data_r10["group"] == g).sum()
            lbl = REGION_EN[g] if is_en else g
            labels_r10.append(f"{lbl}\n(n={n})")

        h, p, n_t = kruskal_groups(data_r10, "group", "value")
        sbox_r10 = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_r10 = "Region" if is_en else "Região do Brasil"
        ylabel_r10 = "Q10 (Work hours sufficiency)" if is_en else "Q10 (Tempo no expediente)"
        fname_r10 = "Q10_region.png" if is_en else "Q10_tempo_expediente_regiao.png"

        plot_boxplot(
            data_r10, "group", "value", order_r10, labels_r10,
            xlabel_r10, ylabel_r10, sbox_r10, (0.98, 0.02, "bottom", "right"),
            out_dir / fname_r10, figsize=(FIG_W, FIG_H_SMALL), mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 8-11. Experience-based (Q14, Q15, Q23, Q25)
        # ────────────────────────────────────────────────────────────────
        exp_configs = [
            ("organizado e estruturado", "Q14",
             "Q14 (Organization and structure)" if is_en else "Q14 (Organização e estrutura)",
             "Q14_experience.png" if is_en else "Q14_organizacao_estrutura.png"),
            ("materiais", "Q15",
             "Q15 (Useful materials)" if is_en else "Q15 (Materiais úteis)",
             "Q15_experience.png" if is_en else "Q15_materiais_uteis.png"),
            ("melhorou meu desempenho", "Q23",
             "Q23 (Performance improvement)" if is_en else "Q23 (Melhora de desempenho)",
             "Q23_experience.png" if is_en else "Q23_melhora_desempenho.png"),
            ("mais autonomia", "Q25",
             "Q25 (Work autonomy)" if is_en else "Q25 (Aumento de autonomia)",
             "Q25_experience.png" if is_en else "Q25_aumento_autonomia.png"),
        ]

        for keyword, qname, ylabel_exp, fname_exp in exp_configs:
            data_exp = prep_experience(df, keyword)
            labels_exp = []
            for g in EXPERIENCIA_ORDER:
                n = (data_exp["group"] == g).sum()
                labels_exp.append(f"{EXP_ABBR[g]}\n(n={n})")

            h, p, n_t = kruskal_groups(data_exp, "group", "value")
            sbox_exp = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

            xlabel_exp = "Total Experience" if is_en else "Tempo Total de Experiência"

            plot_boxplot(
                data_exp, "group", "value", EXPERIENCIA_ORDER, labels_exp,
                xlabel_exp, ylabel_exp, sbox_exp, (0.98, 0.02, "bottom", "right"),
                out_dir / fname_exp, figsize=(FIG_W, FIG_H_MEDIUM), mean_label=mean_lbl,
            )

        # ────────────────────────────────────────────────────────────────
        # 12. Q20 by area
        # ────────────────────────────────────────────────────────────────
        data_a20, order_a20 = prep_area(df, "soft skills")
        labels_a20 = []
        for g in order_a20:
            n = (data_a20["group"] == g).sum()
            lbl = area_lbl_map.get(g, g)
            labels_a20.append(f"{lbl}\n(n={n})")

        h, p, n_t = kruskal_groups(data_a20, "group", "value")
        sbox_a20 = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_a20 = "Professional Area (n={})".format(len(data_a20)) if is_en else "Área de Atuação (n={})".format(len(data_a20))
        ylabel_a20 = "Q20 (Soft skills focus)" if is_en else "Q20 (Foco em Soft Skills)"
        fname_a20 = "Q20_area.png" if is_en else "Q20_foco_soft_skills.png"

        plot_boxplot(
            data_a20, "group", "value", order_a20, labels_a20,
            xlabel_a20, ylabel_a20, sbox_a20, (0.98, 0.05, "bottom", "right"),
            out_dir / fname_a20, figsize=(FIG_W, FIG_H_LARGE),
            x_rotation=90, x_ha="center", legend_loc="upper left",
            mean_label=mean_lbl,
        )

        # ────────────────────────────────────────────────────────────────
        # 13. Q27 by level
        # ────────────────────────────────────────────────────────────────
        data_nv = prep_nivel(df, "oportunidades de crescimento")
        nivel_lbl_map = NIVEL_LABELS_EN if is_en else NIVEL_LABELS_PT
        labels_nv = []
        for g in NIVEL_ORDER:
            n = (data_nv["group"] == g).sum()
            lbl = nivel_lbl_map.get(g, g)
            labels_nv.append(f"{lbl}\n(n={n})")

        h, p, n_t = kruskal_groups(data_nv, "group", "value")
        sbox_nv = stat_box_text(test_name, n_t, h, "H", p, lang) if h else None

        xlabel_nv = "Professional Level" if is_en else "Nível Profissional"
        ylabel_nv = "Q27 (Growth opportunities)" if is_en else "Q27 (Oportunidades de crescimento)"
        fname_nv = "Q27_level.png" if is_en else "Q27_oportunidades_crescimento.png"

        plot_boxplot(
            data_nv, "group", "value", NIVEL_ORDER, labels_nv,
            xlabel_nv, ylabel_nv, sbox_nv, (0.02, 0.98, "top", "left"),
            out_dir / fname_nv, figsize=(FIG_W, FIG_H_MEDIUM), mean_label=mean_lbl,
        )

    # ── Significance heatmap ──────────────────────────────────────────
    generate_significance_heatmap(df, base_out)

    print(f"\nDone. {base_out}/pt/ and {base_out}/en/")


# ── Significance heatmap (from significancia.py logic) ──────────────────
SOCIODEMOGRAPHIC_COLS = {
    'Considerando sua última experiência com essas características, ela foi Obrigatória ou Opcional?': 'Obrig./Opcional',
    'Qual é a sua idade?': 'Idade',
    'Com qual gênero você se identifica?': 'Gênero',
    'Em que Estado você reside? ': 'Estado',
    'Qual é o seu mais alto grau de instrução?': 'Instrução',
    'Qual seu tempo total de experiência na engenharia de software?': 'Experiência',
    'Qual das alternativas abaixo descreve seu nível profissional atual?': 'Nível Prof.',
    'Quantos funcionários a empresa possui? ': 'Tamanho Emp.',
    'Qual é a sua principal área de atuação na engenharia de software no momento? ': 'Área Atuação',
}

SOCIO_LABEL_EN = {
    'Obrig./Opcional': 'Mand./Optional',
    'Idade': 'Age',
    'Gênero': 'Gender',
    'Estado': 'State',
    'Instrução': 'Education',
    'Experiência': 'Experience',
    'Nível Prof.': 'Prof. Level',
    'Tamanho Emp.': 'Company Size',
    'Área Atuação': 'Work Area',
}


def calculate_pvalues(df: pd.DataFrame, likert_cols, socio_cols_found):
    n_questions = len(likert_cols)
    pvalue_matrix = pd.DataFrame(
        np.nan,
        index=[f"Q{i+1}" for i in range(n_questions)],
        columns=[SOCIODEMOGRAPHIC_COLS[col] for col in socio_cols_found],
    )

    for j, socio_col in enumerate(socio_cols_found):
        socio_label = SOCIODEMOGRAPHIC_COLS[socio_col]
        for i, likert_col in enumerate(likert_cols):
            subset = df[[socio_col, likert_col]].copy()
            subset[likert_col] = pd.to_numeric(subset[likert_col], errors="coerce")
            subset = subset.dropna()
            if socio_label == "Experiência":
                subset = subset[subset[socio_col] != "Prefiro não responder"].copy()
            if len(subset) < 5:
                continue

            pval = None
            if socio_label == "Idade":
                try:
                    subset["age_numeric"] = subset[socio_col].apply(convert_age)
                    subset = subset.dropna(subset=["age_numeric"])
                    if len(subset) >= 5:
                        _, pval = stats.spearmanr(subset["age_numeric"], subset[likert_col])
                except Exception:
                    pass
            else:
                try:
                    groups = {}
                    for cat in subset[socio_col].unique():
                        vals = subset[subset[socio_col] == cat][likert_col].values
                        if len(vals) >= 5:
                            groups[cat] = vals
                    if len(groups) >= 2:
                        _, pval = stats.kruskal(*groups.values())
                except Exception:
                    pass

            if pval is not None:
                pvalue_matrix.iloc[i, j] = pval

    return pvalue_matrix


def plot_significance_heatmap(pvalue_matrix: pd.DataFrame, out_path: Path,
                              lang: str = "pt"):
    matrix = pvalue_matrix.copy()
    if lang == "en":
        matrix.columns = [SOCIO_LABEL_EN.get(c, c) for c in matrix.columns]

    n_rows, n_cols = matrix.shape

    # ACM column width, height proportional to rows
    fig_w = FIG_W
    fig_h = max(4.0, 0.22 * n_rows + 0.8)

    # Font sizes for ACM readability
    cell_font = 5
    label_font = 7
    axis_font = 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Build color matrix: white for ns, orange shades for significant
    color_matrix = np.ones((n_rows, n_cols, 4))  # RGBA white
    for i in range(n_rows):
        for j in range(n_cols):
            pval = matrix.iloc[i, j]
            if pd.notna(pval) and pval < 0.05:
                if pval < 0.001:
                    color_matrix[i, j] = [1.0, 0.376, 0.008, 1.0]  # #ff6002
                elif pval < 0.01:
                    color_matrix[i, j] = [1.0, 0.522, 0.2, 1.0]    # #ff8533
                else:
                    color_matrix[i, j] = [1.0, 0.702, 0.4, 1.0]    # #ffb366

    ax.imshow(color_matrix, aspect="auto", interpolation="nearest")

    # Grid lines
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3)

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            pval = matrix.iloc[i, j]
            if pd.isna(pval):
                continue
            if pval < 0.05:
                color = "white"
                weight = "bold"
            else:
                color = "black"
                weight = "normal"
            ax.text(j, i, f"{pval:.4f}", ha="center", va="center",
                    fontsize=cell_font, color=color, fontweight=weight)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, fontsize=label_font, rotation=90, ha="center")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=label_font)

    if lang == "en":
        ax.set_xlabel("Sociodemographic Variables", fontsize=axis_font, fontweight="bold",
                       labelpad=8)
    else:
        ax.set_xlabel("Variáveis Sociodemográficas", fontsize=axis_font, fontweight="bold",
                       labelpad=8)

    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out_path}")


def generate_significance_heatmap(df: pd.DataFrame, base_out: Path):
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    socio_cols_found = [c for c in SOCIODEMOGRAPHIC_COLS.keys() if c in df.columns]

    # Map Likert to numeric
    df_work = df.copy()
    for col in likert_cols:
        df_work[col] = df_work[col].map(LIKERT_MAP)

    pvalue_matrix = calculate_pvalues(df_work, likert_cols, socio_cols_found)

    for lang in ["pt", "en"]:
        out_dir = base_out / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = "significancia_heatmap.png" if lang == "pt" else "significance_heatmap.png"
        plot_significance_heatmap(pvalue_matrix, out_dir / fname, lang=lang)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "tratado.csv"
    out_dir = project_root / "output" / "acm"
    generate_all(csv_path, out_dir)
