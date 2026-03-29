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

GRUPO_ORDER = [
    "Masc. Obrig.",
    "Masc. Opcional",
    "Fem. Obrig.",
    "Fem. Opcional",
]

QUESTION_SHORT = {
    "Q1": "Conexão com objetivos da empresa",
    "Q2": "Apoio e recursos da empresa",
    "Q3": "Conteúdo atendeu necessidades da função",
    "Q4": "Conteúdo adaptado ao nível de experiência",
    "Q5": "Opinião considerada na criação",
    "Q6": "Carga horária prejudicou tempo pessoal",
    "Q7": "Manter-se competitivo no mercado",
    "Q8": "Motivação para buscar conhecimentos",
    "Q9": "Liderança incentivou participação",
    "Q10": "Tempo suficiente no expediente",
    "Q11": "Participou mais por obrigação",
    "Q12": "Incentivos claros da empresa",
    "Q13": "Reconhecimento por concluir treinamentos",
    "Q14": "Bem organizado e estruturado",
    "Q15": "Materiais úteis",
    "Q16": "Ambiente adequado para aprendizado",
    "Q17": "Explicação do instrutor clara",
    "Q18": "Atuação do instrutor fundamental",
    "Q19": "Atividades interessantes e variadas",
    "Q20": "Parte dedicada a soft skills",
    "Q21": "Satisfação com a qualidade",
    "Q22": "Raciocínio para resolver problemas",
    "Q23": "Melhorou desempenho",
    "Q24": "Consigo aplicar no trabalho",
    "Q25": "Mais autonomia no trabalho",
    "Q26": "Suporte da liderança para aplicar",
    "Q27": "Oportunidades de crescimento na empresa",
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
        return "Obrig."
    if "opcion" in t or "opcional" in t:
        return "Opcional"
    return None


def load_and_prepare(csv_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Carrega CSV, cria coluna 'grupo' com 4 categorias (Gênero × Obrig/Opcional).
    Filtra apenas Masculino e Feminino, exclui "Prefiro não responder".
    Retorna: (df_clean, likert_cols)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None

    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Encontrar coluna de gênero
    genero_col = None
    for col in df.columns:
        if "gênero" in str(col).lower() or "genero" in str(col).lower():
            genero_col = col
            break

    # Encontrar coluna obrigatório/opcional
    obrig_col = None
    for col in df.columns:
        lc = str(col).lower()
        if "obrig" in lc and ("opcion" in lc or "opcional" in lc):
            obrig_col = col
            break

    if genero_col is None:
        print("❌ Coluna de gênero não encontrada!")
        return None, None
    if obrig_col is None:
        print("❌ Coluna obrigatório/opcional não encontrada!")
        return None, None

    print(f"✓ Coluna gênero: {repr(genero_col)}")
    print(f"✓ Coluna obrig/opcional: {repr(obrig_col)}")

    # Identificar colunas Likert
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    print(f"✓ Colunas Likert detectadas: {len(likert_cols)}")

    if not likert_cols:
        print("❌ Nenhuma coluna Likert encontrada!")
        return None, None

    # Preparar dataframe
    df_clean = df[[genero_col, obrig_col] + likert_cols].copy()
    df_clean = df_clean.rename(columns={genero_col: "genero", obrig_col: "tipo"})

    # Filtrar apenas Masculino e Feminino
    df_clean = df_clean[df_clean["genero"].isin(["Masculino", "Feminino"])].copy()

    # Mapear obrigatório/opcional
    df_clean["tipo"] = df_clean["tipo"].apply(_map_obrig_opcional)
    df_clean = df_clean.dropna(subset=["tipo"])

    # Criar coluna de grupo
    genero_abrev = {"Masculino": "Masc.", "Feminino": "Fem."}
    df_clean["grupo"] = df_clean["genero"].map(genero_abrev) + " " + df_clean["tipo"]

    # Mapear Likert para numérico
    for col in likert_cols:
        df_clean[col] = df_clean[col].map(LIKERT_MAP)
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    print(f"\n📊 Distribuição dos 4 grupos:")
    for grupo in GRUPO_ORDER:
        n = (df_clean["grupo"] == grupo).sum()
        print(f"   {grupo}: {n}")

    print(f"\n✓ Total: {len(df_clean)} respondentes")

    return df_clean, likert_cols


def run_kruskal_all(df_clean: pd.DataFrame,
                    likert_cols: List[str]) -> pd.DataFrame:
    """
    Executa Kruskal-Wallis para cada questão Likert entre os 4 grupos.
    Retorna DataFrame com H, p-valor, significância e médias por grupo.
    """
    results = []

    for i, col in enumerate(likert_cols):
        q_label = f"Q{i + 1}"
        subset = df_clean[["grupo", col]].dropna()

        # Montar grupos filtrando n >= 5
        grupos = {}
        for grupo in GRUPO_ORDER:
            vals = subset[subset["grupo"] == grupo][col].values
            if len(vals) >= 5:
                grupos[grupo] = vals

        row = {
            "questao": q_label,
            "coluna": col,
        }

        # Adicionar n e média de cada grupo
        for grupo in GRUPO_ORDER:
            key_n = f"n_{grupo}"
            key_m = f"media_{grupo}"
            vals = subset[subset["grupo"] == grupo][col].values
            row[key_n] = len(vals)
            row[key_m] = vals.mean() if len(vals) > 0 else np.nan

        if len(grupos) < 2:
            row["H"] = np.nan
            row["p_valor"] = np.nan
            row["significancia"] = "n/a"
            results.append(row)
            continue

        h_stat, p_valor = stats.kruskal(*grupos.values())

        if p_valor < 0.001:
            sig = "***"
        elif p_valor < 0.01:
            sig = "**"
        elif p_valor < 0.05:
            sig = "*"
        else:
            sig = "ns"

        row["H"] = h_stat
        row["p_valor"] = p_valor
        row["significancia"] = sig
        results.append(row)

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
    ax.set_title("Kruskal-Wallis: Gênero × Obrigatório/Opcional (4 grupos)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim(0, max(p_valores.max() * 1.3, 0.1))
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = out_dir / "kruskal_wallis_genero_obrig_resumo.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n✅ Gráfico resumo salvo em: {out_path}")
    return out_path


def plot_boxplot_significativo(df_clean: pd.DataFrame, col: str,
                               q_label: str, out_dir: Path) -> Optional[Path]:
    """
    Boxplot individual para uma questão significativa (p < 0.05).
    4 caixas: Masc. Obrig., Masc. Opcional, Fem. Obrig., Fem. Opcional.
    """
    if plt is None or sns is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    subset = df_clean[["grupo", col]].dropna()
    orange_palette = _orange_shades(BASE_ORANGE, len(GRUPO_ORDER))
    color_dict = {g: orange_palette[i] for i, g in enumerate(GRUPO_ORDER)}

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.boxplot(
        data=subset,
        x="grupo",
        y=col,
        order=GRUPO_ORDER,
        hue="grupo",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )

    # Médias como bolinhas pretas
    means = subset.groupby("grupo")[col].mean()
    for pos, grupo in enumerate(GRUPO_ORDER):
        if grupo not in means.index:
            continue
        mean_val = means[grupo]
        ax.scatter(pos, mean_val, color="black", s=150, zorder=3,
                   marker="o", edgecolor="white", linewidth=2)

        text = ax.text(pos, mean_val + 0.15, f"{mean_val:.2f}",
                       ha="center", va="bottom", fontsize=18,
                       fontweight="bold", color="black")
        if path_effects is not None:
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal()
            ])

    # Rótulos com n
    group_labels = []
    for grupo in GRUPO_ORDER:
        n = (subset["grupo"] == grupo).sum()
        group_labels.append(f"{grupo}\n(n={n})")

    ax.set_xticks(range(len(GRUPO_ORDER)))
    ax.set_xticklabels(group_labels, fontsize=14)

    # Kruskal-Wallis para o quadro
    grupos = {}
    for grupo in GRUPO_ORDER:
        vals = subset[subset["grupo"] == grupo][col].values
        if len(vals) >= 5:
            grupos[grupo] = vals

    if len(grupos) >= 2:
        h_stat, p_valor = stats.kruskal(*grupos.values())

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
            f"n = {sum(len(g) for g in grupos.values())}\n"
            f"H = {h_stat:.4f}\n"
            f"p = {p_valor:.6f}\n"
            f"Resultado: {sig_symbol} {sig_text}"
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8,
                     edgecolor="black", linewidth=2)
        ax.text(0.98, 0.02, quadro_text, transform=ax.transAxes,
                fontsize=16, verticalalignment="bottom",
                horizontalalignment="right", fontfamily="monospace", bbox=props)

    ax.set_xlabel("Grupo (Gênero × Tipo)", fontsize=20, fontweight="bold")
    q_short = QUESTION_SHORT.get(q_label, "")
    ax.set_ylabel(f"{q_label} - {q_short}", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    out_path = out_dir / f"kruskal_genero_obrig_{q_label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"   ✅ Boxplot salvo: {out_path}")
    return out_path


def print_ranking(results_df: pd.DataFrame) -> None:
    """Imprime ranking de diferenças significativas."""
    print("\n" + "=" * 80)
    print("📊 RANKING DE SIGNIFICÂNCIA - GÊNERO × OBRIGATÓRIO/OPCIONAL (4 grupos)")
    print("=" * 80)

    sig = results_df[results_df["p_valor"] < 0.05].sort_values("p_valor")

    if sig.empty:
        print("❌ Nenhuma diferença estatisticamente significativa encontrada (α=0.05)")
    else:
        print(f"\n🎯 Total de questões significativas: {len(sig)} / {len(results_df)}\n")
        for rank, (_, row) in enumerate(sig.iterrows(), 1):
            medias = " | ".join(
                f"{g}={row.get(f'media_{g}', np.nan):.2f}"
                for g in GRUPO_ORDER
                if pd.notna(row.get(f"media_{g}"))
            )
            print(f"{rank:2d}. {row['questao']:3s} | "
                  f"p={row['p_valor']:.6f} {row['significancia']:3s} | {medias}")

    ns = results_df[results_df["p_valor"] >= 0.05].sort_values("p_valor")
    if not ns.empty:
        print(f"\n--- Questões sem diferença significativa ({len(ns)}) ---")
        for _, row in ns.iterrows():
            print(f"   {row['questao']:3s} | p={row['p_valor']:.6f} ns")

    print("\n" + "=" * 80)


def analise_kruskal_genero_obrig(csv_path: Optional[Path] = None,
                                  out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: Kruskal-Wallis para todas as questões Likert
    entre 4 grupos (Gênero × Obrigatório/Opcional).
    """
    project_root = Path(__file__).resolve().parents[2]

    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "tratado.csv"
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output" / "kruskal_genero_obrig"

    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return {"error": f"CSV não encontrado: {csv_path}"}

    print(f"📂 Carregando: {csv_path}\n")

    # 1. Carregar e preparar dados
    df_clean, likert_cols = load_and_prepare(csv_path)
    if df_clean is None:
        return {"error": "Falha ao preparar dados"}

    # 2. Executar Kruskal-Wallis para todas as questões
    print(f"\n🔬 Executando Kruskal-Wallis (4 grupos) para {len(likert_cols)} questões...")
    results_df = run_kruskal_all(df_clean, likert_cols)

    # 3. Gráfico resumo
    summary_path = plot_summary(results_df, out_dir)

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
        "boxplot_paths": boxplot_paths,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None

    result = analise_kruskal_genero_obrig(csv_path=csv_arg, out_dir=out_arg)

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
