"""
Validação estatística da Curva em U (Experiência × Percepção).

Para cada item Likert (Q14, Q15, Q23, Q25):
  1. Regressão polinomial (linear vs. quadrática) com ANOVA de modelos aninhados
  2. Teste de tendência (Kendall tau como proxy de Jonckheere-Terpstra)
  3. Visualização com IC 95% e curvas ajustadas
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_ORANGE = "#ff6002"

LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

EXPERIENCE_COL = "Qual seu tempo total de experiência na engenharia de software?"
EXPERIENCE_MAP = {
    "Até um ano.": 0.5,
    "Entre 1 e 2 anos.": 1.5,
    "Entre 3 e 4 anos.": 3.5,
    "Entre 5 e 6 anos.": 5.5,
    "Entre 7 e 8 anos.": 7.5,
    "Mais de 8 anos.": 10.0,
}

EXPERIENCE_ORDER = [
    "Até um ano.",
    "Entre 1 e 2 anos.",
    "Entre 3 e 4 anos.",
    "Entre 5 e 6 anos.",
    "Entre 7 e 8 anos.",
    "Mais de 8 anos.",
]

# Q14, Q15, Q23, Q25 (0-indexed among Likert columns)
ITEMS = {
    "Q14": {"idx": 13, "label": "Organização e estrutura"},
    "Q15": {"idx": 14, "label": "Materiais úteis"},
    "Q23": {"idx": 22, "label": "Melhora de desempenho"},
    "Q25": {"idx": 24, "label": "Autonomia no trabalho"},
}

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tratado.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "curva_u"


# ---------------------------------------------------------------------------
# Carregamento e preparação
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, dtype=str, engine="python")

    # Identificar colunas Likert
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    print(f"Colunas Likert detectadas: {len(likert_cols)}")

    # Encontrar coluna de experiência (busca flexível)
    exp_col = None
    for c in df.columns:
        if "tempo total de experiência" in c.lower():
            exp_col = c
            break
    if exp_col is None:
        raise ValueError("Coluna de experiência não encontrada")

    # Mapear Likert para numérico
    df_likert = df[likert_cols].copy()
    for col in df_likert.columns:
        df_likert[col] = df_likert[col].map(LIKERT_MAP)
    df_likert = df_likert.apply(pd.to_numeric, errors="coerce")

    # Experiência
    df["exp_str"] = df[exp_col].str.strip()
    df["exp_num"] = df["exp_str"].map(EXPERIENCE_MAP)

    # Combinar
    result = pd.concat([df[["exp_str", "exp_num"]], df_likert], axis=1)
    result = result.dropna(subset=["exp_num"])

    return result, likert_cols


# ---------------------------------------------------------------------------
# Análise de regressão para um item
# ---------------------------------------------------------------------------
def analyze_item(df, likert_cols, item_key):
    info = ITEMS[item_key]
    col = likert_cols[info["idx"]]
    data = df[["exp_str", "exp_num", col]].dropna().copy()
    data.columns = ["exp_str", "x", "y"]

    # --- Estatísticas descritivas por faixa ---
    group_stats = []
    for grp in EXPERIENCE_ORDER:
        subset = data[data["exp_str"] == grp]["y"]
        if len(subset) == 0:
            continue
        n = len(subset)
        mean = subset.mean()
        std = subset.std(ddof=1)
        se = std / np.sqrt(n)
        ci95 = 1.96 * se
        group_stats.append({
            "faixa": grp,
            "x_mid": EXPERIENCE_MAP[grp],
            "n": n,
            "mean": mean,
            "std": std,
            "ci95": ci95,
        })
    gs = pd.DataFrame(group_stats)

    # --- Modelo linear: y ~ x ---
    X_lin = sm.add_constant(data["x"])
    model_lin = sm.OLS(data["y"], X_lin).fit()

    # --- Modelo quadrático: y ~ x + x² ---
    X_quad = sm.add_constant(np.column_stack([data["x"], data["x"] ** 2]))
    model_quad = sm.OLS(data["y"], X_quad).fit()

    # --- Kruskal-Wallis (diferença global entre faixas) ---
    groups = [data[data["exp_str"] == grp]["y"].values for grp in EXPERIENCE_ORDER
              if len(data[data["exp_str"] == grp]) > 0]
    kw_stat, kw_p = stats.kruskal(*groups)

    # β₂
    beta2_coef = model_quad.params[2]
    beta2_pval = model_quad.pvalues[2]

    # --- Kendall tau (proxy Jonckheere-Terpstra) ---
    # Converter faixas para ranks ordinais (1..6)
    rank_map = {grp: i + 1 for i, grp in enumerate(EXPERIENCE_ORDER)}
    data["rank"] = data["exp_str"].map(rank_map)
    tau, tau_p = stats.kendalltau(data["rank"], data["y"])

    results = {
        "item": item_key,
        "label": info["label"],
        "R2_linear": model_lin.rsquared,
        "F_linear": model_lin.fvalue,
        "p_linear": model_lin.f_pvalue,
        "R2_quadratic": model_quad.rsquared,
        "F_quadratic": model_quad.fvalue,
        "p_quadratic": model_quad.f_pvalue,
        "beta2_coef": beta2_coef,
        "beta2_pval": beta2_pval,
        "kruskal_H": kw_stat,
        "kruskal_p": kw_p,
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "conclusion": "U-shape confirmed" if beta2_pval < 0.05 else "U-shape not confirmed",
    }

    return results, gs, model_lin, model_quad


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
def plot_item(item_key, gs, model_lin, model_quad, results):
    # ACM single-column: 3.33in width, ~2.5in height, 300 dpi
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, ax = plt.subplots(figsize=(3.33, 3.0))

    x_mid = gs["x_mid"].values
    means = gs["mean"].values
    ci95 = gs["ci95"].values
    ns = gs["n"].astype(int).values

    # Pontos com barras de erro
    colors = ["#d62728" if n < 30 else BASE_ORANGE for n in ns]
    ax.errorbar(x_mid, means, yerr=ci95, fmt="none", capsize=3, color="gray",
                zorder=2, linewidth=0.8)
    ax.scatter(x_mid, means, c=colors, s=35, zorder=3, edgecolors="black", linewidths=0.4)

    # Curva linear
    x_smooth = np.linspace(0, 11, 200)
    coef_lin = np.array([model_lin.params[1], model_lin.params[0]])
    ax.plot(
        x_smooth,
        np.polyval(coef_lin, x_smooth),
        "--",
        color="steelblue",
        alpha=0.7,
        linewidth=1,
        label=f"Linear (R²={results['R2_linear']:.4f})",
    )

    # Curva quadrática
    coef_quad = np.array([model_quad.params[2], model_quad.params[1], model_quad.params[0]])
    ax.plot(
        x_smooth,
        np.polyval(coef_quad, x_smooth),
        "-",
        color="#d62728",
        linewidth=1.2,
        label=f"Quad. (R²={results['R2_quadratic']:.4f}, p(β₂)={results['beta2_pval']:.4f})",
    )

    # Anotações n por faixa — alternar acima/abaixo para evitar sobreposição
    for i, (x, n, m) in enumerate(zip(x_mid, ns, means)):
        color = "#d62728" if n < 30 else "black"
        offset_y = 10 if i % 2 == 0 else -14
        va = "bottom" if offset_y > 0 else "top"
        ax.annotate(f"n={n}", (x, m), textcoords="offset points",
                    xytext=(0, offset_y), ha="center", va=va,
                    fontsize=6, color=color)

    # Labels das faixas no eixo x
    faixa_labels = ["≤1", "1-2", "3-4", "5-6", "7-8", ">8"]
    ax.set_xticks(list(EXPERIENCE_MAP.values()))
    ax.set_xticklabels(faixa_labels)

    ax.set_xlabel("Tempo de Experiência (anos)")
    ax.set_ylabel("Média Likert")
    ax.set_title(f"{item_key} — {results['label']}")
    ax.set_ylim(1.5, 5.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.4)

    # Texto com resultados — posicionar no canto inferior esquerdo, sem sobrepor legenda
    txt = (
        f"KW: H={results['kruskal_H']:.2f}, p={results['kruskal_p']:.4f}\n"
        f"τ={results['kendall_tau']:.3f}, p={results['kendall_p']:.4f}"
    )
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=5.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

    plt.tight_layout(pad=0.4)
    out_path = OUTPUT_DIR / f"{item_key}_curva_u.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico salvo: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, likert_cols = load_data()
    print(f"Registros com experiência válida: {len(df)}\n")

    all_results = []

    for item_key in ITEMS:
        results, gs, model_lin, model_quad = analyze_item(df, likert_cols, item_key)
        all_results.append(results)

        # Print formatado
        r = results
        print(f"{r['item']} ({r['label']})")
        print(f"  Modelo linear:     R² = {r['R2_linear']:.4f}, F = {r['F_linear']:.2f}, p = {r['p_linear']:.4f}")
        print(f"  Modelo quadrático: R² = {r['R2_quadratic']:.4f}, F = {r['F_quadratic']:.2f}, p = {r['p_quadratic']:.4f}")
        print(f"  β₂ (termo quadrático): coef = {r['beta2_coef']:.4f}, p = {r['beta2_pval']:.4f}")
        print(f"  Kruskal-Wallis: H = {r['kruskal_H']:.2f}, p = {r['kruskal_p']:.4f}")
        print(f"  Kendall τ (tendência): tau = {r['kendall_tau']:.3f}, p = {r['kendall_p']:.4f}")
        print(f"  → {r['conclusion']}")
        print()

        # Gráfico
        plot_item(item_key, gs, model_lin, model_quad, results)

    # Salvar CSV
    df_results = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "regression_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")


if __name__ == "__main__":
    main()
