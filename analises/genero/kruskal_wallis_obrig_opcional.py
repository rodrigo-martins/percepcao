from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as path_effects
    import seaborn as sns
except Exception:
    plt = None
    sns = None
    mcolors = None
    path_effects = None

from scipy import stats

BASE_ORANGE = "#ff6002"

LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

TIPO_ORDER = ["Obrigatória", "Opcional"]

QUESTION_SHORT = {
    "Q1": "Conexão com objetivos",
    "Q2": "Apoio e recursos",
    "Q3": "Relevância para a função",
    "Q4": "Adaptação ao nível",
    "Q5": "Consideração da opinião",
    "Q6": "Prejuízo ao tempo pessoal",
    "Q7": "Competitividade no mercado",
    "Q8": "Motivação para aprender",
    "Q9": "Incentivo da liderança",
    "Q10": "Tempo no expediente",
    "Q11": "Obrigação acima do interesse",
    "Q12": "Incentivos tangíveis",
    "Q13": "Reconhecimento",
    "Q14": "Organização e estrutura",
    "Q15": "Materiais úteis",
    "Q16": "Ambiente adequado",
    "Q17": "Clareza do instrutor",
    "Q18": "Atuação do instrutor",
    "Q19": "Atividades variadas",
    "Q20": "Foco em Soft Skills",
    "Q21": "Satisfação geral",
    "Q22": "Resolução de problemas",
    "Q23": "Melhora de desempenho",
    "Q24": "Aplicabilidade prática",
    "Q25": "Aumento de autonomia",
    "Q26": "Suporte à aplicação",
    "Q27": "Oportunidades de crescimento",
}


def _orange_shades(base_hex: str, n: int) -> list:
    """Gera n tons do laranja base: índice 0 -> escuro (base), índice n-1 -> claro."""
    if mcolors is None or np is None:
        return [base_hex] * max(1, n)
    base_rgb = np.array(mcolors.to_rgb(base_hex))
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


def _map_obrig_opcional(text: str) -> Optional[str]:
    """Mapeia texto da coluna obrigatório/opcional para categoria curta."""
    t = str(text).strip().lower()
    if "obrig" in t:
        return "Obrigatória"
    if "opcion" in t or "opcional" in t:
        return "Opcional"
    return None


def load_and_prepare(csv_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Carrega CSV, identifica coluna obrigatório/opcional e colunas Likert.
    Retorna: (df_clean, likert_cols)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None

    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Encontrar coluna obrigatório/opcional
    obrig_col = None
    for col in df.columns:
        lc = str(col).lower()
        if "obrig" in lc and ("opcion" in lc or "opcional" in lc):
            obrig_col = col
            break

    if obrig_col is None:
        print("❌ Coluna obrigatório/opcional não encontrada!")
        return None, None

    print(f"✓ Coluna obrig/opcional: {repr(obrig_col)}")

    # Identificar colunas Likert
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    print(f"✓ Colunas Likert detectadas: {len(likert_cols)}")

    if not likert_cols:
        print("❌ Nenhuma coluna Likert encontrada!")
        return None, None

    # Preparar dataframe
    df_clean = df[[obrig_col] + likert_cols].copy()
    df_clean = df_clean.rename(columns={obrig_col: "tipo"})

    # Mapear obrigatório/opcional
    df_clean["tipo"] = df_clean["tipo"].apply(_map_obrig_opcional)

    print(f"\n📊 Distribuição (antes do filtro):")
    for tipo, count in df_clean["tipo"].value_counts().items():
        print(f"   {tipo}: {count}")

    df_clean = df_clean.dropna(subset=["tipo"])

    # Mapear Likert para numérico
    for col in likert_cols:
        df_clean[col] = df_clean[col].map(LIKERT_MAP)
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print(f"\n✓ Total: {len(df_clean)} respondentes")
    for tipo in TIPO_ORDER:
        n = (df_clean["tipo"] == tipo).sum()
        print(f"   {tipo}: {n}")

    return df_clean, likert_cols


def run_kruskal_all(df_clean: pd.DataFrame,
                    likert_cols: List[str]) -> pd.DataFrame:
    """
    Executa Kruskal-Wallis para cada questão Likert entre Obrigatória e Opcional.
    """
    results = []

    for i, col in enumerate(likert_cols):
        q_label = f"Q{i + 1}"
        subset = df_clean[["tipo", col]].dropna()

        grupo_obrig = subset[subset["tipo"] == "Obrigatória"][col].values
        grupo_opcio = subset[subset["tipo"] == "Opcional"][col].values

        if len(grupo_obrig) < 5 or len(grupo_opcio) < 5:
            results.append({
                "questao": q_label,
                "coluna": col,
                "n_obrig": len(grupo_obrig),
                "n_opcional": len(grupo_opcio),
                "media_obrig": np.nan,
                "media_opcional": np.nan,
                "H": np.nan,
                "p_valor": np.nan,
                "significancia": "n/a",
            })
            continue

        h_stat, p_valor = stats.kruskal(grupo_obrig, grupo_opcio)

        if p_valor < 0.001:
            sig = "***"
        elif p_valor < 0.01:
            sig = "**"
        elif p_valor < 0.05:
            sig = "*"
        else:
            sig = "ns"

        results.append({
            "questao": q_label,
            "coluna": col,
            "n_obrig": len(grupo_obrig),
            "n_opcional": len(grupo_opcio),
            "media_obrig": grupo_obrig.mean(),
            "media_opcional": grupo_opcio.mean(),
            "H": h_stat,
            "p_valor": p_valor,
            "significancia": sig,
        })

    return pd.DataFrame(results)


def plot_summary(results_df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Gráfico resumo horizontal: barras de p-valores para cada questão,
    com linha de corte em α = 0.05 e destaque laranja para significativos.
    """
    if plt is None:
        print("❌ matplotlib não disponível")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = results_df.dropna(subset=["p_valor"]).copy()
    df_plot = df_plot.sort_values("questao", key=lambda x: x.str.extract(r"(\d+)")[0].astype(int))

    fig, ax = plt.subplots(figsize=(16, 12))

    questoes = df_plot["questao"].values
    questoes_labels = [f"{q} - {QUESTION_SHORT.get(q, '')}" for q in questoes]
    p_valores = df_plot["p_valor"].values
    y_pos = np.arange(len(questoes))

    cores = [BASE_ORANGE if p < 0.05 else "#d0d0d0" for p in p_valores]

    ax.barh(y_pos, p_valores, color=cores, edgecolor="white", height=0.7)

    ax.axvline(x=0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05")

    for idx, (p, sig) in enumerate(zip(p_valores, df_plot["significancia"].values)):
        x_text = p + 0.005
        label = f"p = {p:.4f} {sig}"
        fontweight = "bold" if p < 0.05 else "normal"
        color = "black" if p < 0.05 else "gray"
        ax.text(x_text, idx, label, va="center", fontsize=11,
                fontweight=fontweight, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(questoes_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("p-valor (Kruskal-Wallis)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Questões Likert", fontsize=14, fontweight="bold")
    ax.set_title("Kruskal-Wallis: Obrigatória vs Opcional", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim(0, max(p_valores.max() * 1.3, 0.1))
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = out_dir / "kruskal_wallis_obrig_opcional_resumo.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n✅ Gráfico resumo salvo em: {out_path}")
    return out_path


def plot_table_heatmap(results_df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Tabela-heatmap com questão (short_label), p-valor, média obrigatória, média opcional.
    Ordenada pela maior diferença |Opcional - Obrigatória|.
    Escala de laranja no p-valor (padrão significancia.py).
    """
    if plt is None:
        print("❌ matplotlib não disponível")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    df_tab = results_df.dropna(subset=["p_valor"]).copy()
    df_tab["short"] = df_tab["questao"].map(QUESTION_SHORT)
    df_tab["label"] = df_tab["questao"] + " - " + df_tab["short"]
    df_tab["diff"] = (df_tab["media_opcional"] - df_tab["media_obrig"]).abs()
    df_tab = df_tab.sort_values("diff", ascending=False).reset_index(drop=True)

    n_rows = len(df_tab)
    col_labels = ["Questão", "p-valor", "Obrigatória", "Opcional"]
    n_cols = len(col_labels)

    # Dimensões
    row_h = 0.45
    fig_h = max(4, 1.2 + n_rows * row_h)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    # Larguras das colunas (proporcionais)
    col_widths = [5.5, 2.5, 3.0, 3.0]
    col_starts = [0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)
    # Normalizar para caber em n_cols
    total_w = sum(col_widths)
    col_starts = [s / total_w * n_cols for s in col_starts]
    col_widths_norm = [w / total_w * n_cols for w in col_widths]

    # Cabeçalho
    header_y = n_rows + 0.5
    for j, label in enumerate(col_labels):
        x_center = col_starts[j] + col_widths_norm[j] / 2
        ax.text(x_center, header_y, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="black")

    # Linha separadora do cabeçalho
    ax.plot([0, n_cols], [n_rows, n_rows], color="black", linewidth=2)

    # Linhas de dados
    for i, (_, row) in enumerate(df_tab.iterrows()):
        y = n_rows - 1 - i + 0.5
        pval = row["p_valor"]

        # Cor de fundo para a linha inteira baseada no p-valor
        if pval < 0.001:
            bg_color = BASE_ORANGE       # #ff6002
            text_color = "white"
        elif pval < 0.01:
            bg_color = "#ff8533"          # laranja médio
            text_color = "white"
        elif pval < 0.05:
            bg_color = "#ffb366"          # laranja claro
            text_color = "black"
        else:
            bg_color = "white"
            text_color = "black"

        # Fundo da linha
        rect = plt.Rectangle((0, y - 0.5), n_cols, 1, facecolor=bg_color,
                              edgecolor="gray", linewidth=0.5)
        ax.add_patch(rect)

        # Valores das células
        cell_values = [
            row["label"],
            f"{pval:.4f}",
            f"{row['media_obrig']:.2f}",
            f"{row['media_opcional']:.2f}",
        ]
        fontweights = [
            "bold" if pval < 0.05 else "normal",
            "bold" if pval < 0.05 else "normal",
            "normal",
            "normal",
        ]
        aligns = ["left", "center", "center", "center"]

        for j, (val, fw, ha) in enumerate(zip(cell_values, fontweights, aligns)):
            if ha == "left":
                x = col_starts[j] + 0.1
            else:
                x = col_starts[j] + col_widths_norm[j] / 2
            ax.text(x, y, val, ha=ha, va="center",
                    fontsize=11, fontweight=fw, color=text_color)

    # Borda inferior
    ax.plot([0, n_cols], [0, 0], color="black", linewidth=1)

    plt.tight_layout()
    out_path = out_dir / "tabela_obrig_opcional.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n✅ Tabela heatmap salva em: {out_path}")
    return out_path


def plot_boxplot_significativo(df_clean: pd.DataFrame, col: str,
                               q_label: str, out_dir: Path) -> Optional[Path]:
    """
    Boxplot individual para uma questão significativa (p < 0.05).
    """
    if plt is None or sns is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    subset = df_clean[["tipo", col]].dropna()
    orange_palette = _orange_shades(BASE_ORANGE, len(TIPO_ORDER))
    color_dict = {t: orange_palette[i] for i, t in enumerate(TIPO_ORDER)}

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(
        data=subset,
        x="tipo",
        y=col,
        order=TIPO_ORDER,
        hue="tipo",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.5,
        linewidth=2.5,
    )

    # Médias como bolinhas pretas
    means = subset.groupby("tipo")[col].mean()
    for pos, tipo in enumerate(TIPO_ORDER):
        mean_val = means[tipo]
        ax.scatter(pos, mean_val, color="black", s=150, zorder=3,
                   marker="o", edgecolor="white", linewidth=2)

        text = ax.text(pos, mean_val + 0.15, f"{mean_val:.2f}",
                       ha="center", va="bottom", fontsize=20,
                       fontweight="bold", color="black")
        if path_effects is not None:
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal()
            ])

    # Rótulos com n
    group_labels = []
    for tipo in TIPO_ORDER:
        n = (subset["tipo"] == tipo).sum()
        group_labels.append(f"{tipo}\n(n={n})")

    ax.set_xticks(range(len(TIPO_ORDER)))
    ax.set_xticklabels(group_labels, fontsize=16)

    # Kruskal-Wallis para o quadro
    grupo_obrig = subset[subset["tipo"] == "Obrigatória"][col].values
    grupo_opcio = subset[subset["tipo"] == "Opcional"][col].values
    h_stat, p_valor = stats.kruskal(grupo_obrig, grupo_opcio)

    if p_valor < 0.001:
        sig_symbol = "***"
        sig_text = "Altamente significativo"
    elif p_valor < 0.01:
        sig_symbol = "**"
        sig_text = "Muito significativo"
    elif p_valor < 0.05:
        sig_symbol = "*"
        sig_text = "Significativo"
    else:
        sig_symbol = "ns"
        sig_text = "Não significativo"

    quadro_text = (
        f"Teste de Kruskal-Wallis\n"
        f"{'─' * 32}\n"
        f"n = {len(grupo_obrig) + len(grupo_opcio)}\n"
        f"H = {h_stat:.4f}\n"
        f"p = {p_valor:.6f}\n"
        f"Resultado: {sig_symbol} {sig_text}"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8,
                 edgecolor="black", linewidth=2)
    ax.text(0.98, 0.02, quadro_text, transform=ax.transAxes,
            fontsize=16, verticalalignment="bottom",
            horizontalalignment="right", fontfamily="monospace", bbox=props)

    ax.set_xlabel("Tipo de Experiência", fontsize=20, fontweight="bold")
    q_short = QUESTION_SHORT.get(q_label, "")
    ax.set_ylabel(f"{q_label} - {q_short}", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    out_path = out_dir / f"kruskal_obrig_opcional_{q_label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"   ✅ Boxplot salvo: {out_path}")
    return out_path


def print_ranking(results_df: pd.DataFrame) -> None:
    """Imprime ranking de diferenças significativas."""
    print("\n" + "=" * 70)
    print("📊 RANKING DE SIGNIFICÂNCIA - OBRIGATÓRIA vs OPCIONAL")
    print("=" * 70)

    sig = results_df[results_df["p_valor"] < 0.05].sort_values("p_valor")

    if sig.empty:
        print("❌ Nenhuma diferença estatisticamente significativa encontrada (α=0.05)")
    else:
        print(f"\n🎯 Total de questões significativas: {len(sig)} / {len(results_df)}\n")
        for rank, (_, row) in enumerate(sig.iterrows(), 1):
            print(f"{rank:2d}. {row['questao']:3s} | "
                  f"p={row['p_valor']:.6f} {row['significancia']:3s} | "
                  f"Obrig={row['media_obrig']:.2f} Opcional={row['media_opcional']:.2f}")

    ns = results_df[results_df["p_valor"] >= 0.05].sort_values("p_valor")
    if not ns.empty:
        print(f"\n--- Questões sem diferença significativa ({len(ns)}) ---")
        for _, row in ns.iterrows():
            print(f"   {row['questao']:3s} | p={row['p_valor']:.6f} ns")

    print("\n" + "=" * 70)


def export_results_csv(results_df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Exporta resultados do Kruskal-Wallis para CSV reutilizável."""
    out_dir.mkdir(parents=True, exist_ok=True)

    export_df = results_df.copy()
    export_df["short_label"] = export_df["questao"].map(QUESTION_SHORT)
    export_df["diff_opcional_obrig"] = export_df["media_opcional"] - export_df["media_obrig"]

    # Reordenar colunas para facilitar leitura
    cols = [
        "questao", "short_label", "n_obrig", "n_opcional",
        "media_obrig", "media_opcional", "diff_opcional_obrig",
        "H", "p_valor", "significancia",
    ]
    export_df = export_df[cols].sort_values("diff_opcional_obrig", key=abs, ascending=False)

    csv_path = out_dir / "resultados_obrig_opcional.csv"
    export_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n💾 Resultados exportados: {csv_path}")
    return csv_path


def analise_kruskal_obrig_opcional(csv_path: Optional[Path] = None,
                                    out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: Kruskal-Wallis para todas as questões Likert
    entre Obrigatória e Opcional.
    """
    project_root = Path(__file__).resolve().parents[2]

    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "tratado.csv"
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output" / "kruskal_obrig_opcional"

    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return {"error": f"CSV não encontrado: {csv_path}"}

    print(f"📂 Carregando: {csv_path}\n")

    # 1. Carregar e preparar dados
    df_clean, likert_cols = load_and_prepare(csv_path)
    if df_clean is None:
        return {"error": "Falha ao preparar dados"}

    # 2. Executar Kruskal-Wallis para todas as questões
    print(f"\n🔬 Executando Kruskal-Wallis para {len(likert_cols)} questões...")
    results_df = run_kruskal_all(df_clean, likert_cols)

    # 3. Gráfico resumo
    summary_path = plot_summary(results_df, out_dir)

    # 3b. Tabela heatmap
    table_path = plot_table_heatmap(results_df, out_dir)

    # 3c. Exportar resultados para CSV
    csv_out = export_results_csv(results_df, out_dir)

    # 4. Boxplots para questões significativas
    sig_questions = results_df[results_df["p_valor"] < 0.05]
    boxplot_paths = []

    if not sig_questions.empty:
        print(f"\n📈 Gerando boxplots para {len(sig_questions)} questões significativas...")
        for _, row in sig_questions.iterrows():
            bp_path = plot_boxplot_significativo(
                df_clean, row["coluna"], row["questao"], out_dir
            )
            if bp_path:
                boxplot_paths.append(bp_path)
    else:
        print("\n⚠️ Nenhuma questão significativa — nenhum boxplot individual gerado.")

    # 5. Ranking
    print_ranking(results_df)

    return {
        "status": "ok",
        "n_questions": len(likert_cols),
        "n_significativas": len(sig_questions),
        "results_df": results_df,
        "summary_path": summary_path,
        "table_path": table_path,
        "csv_path": csv_out,
        "boxplot_paths": boxplot_paths,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None

    result = analise_kruskal_obrig_opcional(csv_path=csv_arg, out_dir=out_arg)

    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Questões analisadas: {result['n_questions']}")
        print(f"   Questões significativas: {result['n_significativas']}")
        print(f"   Resumo: {result['summary_path']}")
        if result["boxplot_paths"]:
            print(f"   Boxplots: {len(result['boxplot_paths'])} gerados")
