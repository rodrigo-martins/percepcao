from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

BASE_ORANGE = "#ff6002"

# Tradução dos rótulos sociodemográficos para inglês
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

# Mapeamento exato das colunas sociodemográficas
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

# Mapeamento da escala Likert
LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}


def load_and_preprocess(csv_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Carrega CSV, identifica colunas Likert e sociodemográficas.
    Retorna: (df_processado, likert_cols, socio_cols_found)
    """
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    
    # Identificar colunas Likert (começam com '[')
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    
    # Identificar colunas sociodemográficas presentes no CSV
    socio_cols_found = [c for c in SOCIODEMOGRAPHIC_COLS.keys() if c in df.columns]
    
    print(f"✓ Colunas Likert detectadas: {len(likert_cols)}")
    print(f"✓ Colunas sociodemográficas encontradas: {len(socio_cols_found)}/{len(SOCIODEMOGRAPHIC_COLS)}")
    
    return df, likert_cols, socio_cols_found


def prepare_likert_data(df: pd.DataFrame, likert_cols: List[str]) -> pd.DataFrame:
    """Mapeia respostas Likert para números (1-5)."""
    df_likert = df[likert_cols].copy()
    
    for col in df_likert.columns:
        df_likert[col] = df_likert[col].map(LIKERT_MAP)
    
    # Converter para numérico, remover NaNs
    df_likert = df_likert.apply(pd.to_numeric, errors="coerce")
    df_likert = df_likert.dropna(axis=0, how="any")
    
    return df_likert


def convert_age_to_numeric(age_str: str) -> Optional[float]:
    """Converte string de idade (faixa ou número) para valor numérico."""
    if pd.isna(age_str):
        return None
    
    age_str = str(age_str).strip().lower()
    
    # Mapeamento de faixas de idade
    age_map = {
        "até 20 anos": 20,
        "21 a 30 anos": 25.5,
        "31 a 40 anos": 35.5,
        "41 a 50 anos": 45.5,
        "51 a 60 anos": 55.5,
        "mais de 60 anos": 65,
        "de 18 a 25 anos": 21.5,
        "de 26 a 35 anos": 30.5,
        "de 36 a 45 anos": 40.5,
        "de 46 a 55 anos": 50.5,
        "acima de 55 anos": 60,
    }
    
    for key, val in age_map.items():
        if key in age_str:
            return float(val)
    
    # Tentar extrair número
    import re
    nums = re.findall(r"\d+", age_str)
    if nums:
        return float(nums[0])
    
    return None


def classify_effect_size(value: float, test_type: str) -> str:
    """Classifica o tamanho de efeito em negligível/pequeno/médio/grande."""
    if test_type == "spearman":
        v = abs(value)
        if v < 0.10:
            return "negligível"
        elif v < 0.30:
            return "pequeno"
        elif v < 0.50:
            return "médio"
        else:
            return "grande"
    else:  # epsilon_squared (Kruskal-Wallis)
        if value < 0.01:
            return "negligível"
        elif value < 0.06:
            return "pequeno"
        elif value < 0.14:
            return "médio"
        else:
            return "grande"


def calculate_pvalues(df: pd.DataFrame, likert_cols: List[str],
                     socio_cols_found: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Calcula matriz de p-valores entre demográficos e questões Likert.

    Regras:
    - Idade: Spearman correlation
    - Outros categóricos: Kruskal-Wallis
    - Ignora grupos com n < 5
    - Remove "Prefiro não responder" de Experiência

    Retorna: (pvalue_matrix, details) onde details contém estatísticas
    e tamanho de efeito para cada combinação significativa.
    """
    n_questions = len(likert_cols)
    n_socio = len(socio_cols_found)

    # Criar matriz vazia
    pvalue_matrix = pd.DataFrame(
        np.nan,
        index=[f"Q{i+1}" for i in range(n_questions)],
        columns=[SOCIODEMOGRAPHIC_COLS[col] for col in socio_cols_found]
    )

    details = []

    print("\n🔬 Calculando p-valores...")

    for j, socio_col in enumerate(socio_cols_found):
        socio_label = SOCIODEMOGRAPHIC_COLS[socio_col]

        for i, likert_col in enumerate(likert_cols):
            subset = df[[socio_col, likert_col]].copy()
            subset[likert_col] = pd.to_numeric(subset[likert_col], errors="coerce")
            subset = subset.dropna()

            # REMOVER "Prefiro não responder" de Experiência
            if socio_label == "Experiência":
                subset = subset[subset[socio_col] != "Prefiro não responder"].copy()

            if len(subset) < 5:
                continue

            pval = None
            test_stat = None
            test_type = None
            effect_size = None
            effect_label = None

            # Regra 1: Idade → Spearman
            if socio_label == "Idade":
                try:
                    subset["age_numeric"] = subset[socio_col].apply(convert_age_to_numeric)
                    subset = subset.dropna(subset=["age_numeric"])
                    if len(subset) >= 5:
                        r, pval = stats.spearmanr(subset["age_numeric"], subset[likert_col])
                        test_stat = r
                        test_type = "spearman"
                        effect_size = abs(r)
                        effect_label = classify_effect_size(r, "spearman")
                except Exception:
                    pass

            # Regra 2: Categóricos → Kruskal-Wallis
            else:
                try:
                    groups = {}
                    for cat in subset[socio_col].unique():
                        vals = subset[subset[socio_col] == cat][likert_col].values
                        if len(vals) >= 5:
                            groups[cat] = vals

                    if len(groups) >= 2:
                        H, pval = stats.kruskal(*groups.values())
                        test_stat = H
                        test_type = "kruskal"
                        n = sum(len(v) for v in groups.values())
                        effect_size = H / (n - 1)  # epsilon-squared
                        effect_label = classify_effect_size(effect_size, "kruskal")
                except Exception:
                    pass

            if pval is not None:
                pvalue_matrix.iloc[i, j] = pval

                detail = {
                    "socio_col": socio_col,
                    "socio_label": socio_label,
                    "likert_col": likert_col,
                    "q_label": f"Q{i+1}",
                    "test_type": test_type,
                    "test_stat": test_stat,
                    "p_value": pval,
                    "effect_size": effect_size,
                    "effect_label": effect_label,
                }
                details.append(detail)

                if pval < 0.05:
                    if test_type == "spearman":
                        print(f"  ⭐ {socio_label:15} × Q{i+1:2d}: r={test_stat:.3f}, p={pval:.4f} (efeito {effect_label})")
                    else:
                        print(f"  ⭐ {socio_label:15} × Q{i+1:2d}: H={test_stat:.2f}, p={pval:.4f}, ε²={effect_size:.3f} (efeito {effect_label})")

    return pvalue_matrix, details


def run_posthoc(df: pd.DataFrame, details: List[Dict],
                likert_cols: List[str]) -> List[Dict]:
    """
    Executa testes post-hoc de Dunn com correção de Bonferroni para
    combinações Kruskal-Wallis significativas (p < 0.05).

    Retorna lista de resultados post-hoc com pares significativos.
    """
    if sp is None:
        print("\n⚠️  scikit-posthocs não instalado. Instale com: pip install scikit-posthocs")
        return []

    significant_kw = [d for d in details if d["test_type"] == "kruskal" and d["p_value"] < 0.05]

    if not significant_kw:
        print("\n📊 Nenhum Kruskal-Wallis significativo para post-hoc.")
        return []

    print(f"\n🔍 Executando post-hoc de Dunn (Bonferroni) para {len(significant_kw)} combinações...")

    posthoc_results = []

    for d in significant_kw:
        socio_col = d["socio_col"]
        socio_label = d["socio_label"]
        likert_col = d["likert_col"]
        q_label = d["q_label"]

        subset = df[[socio_col, likert_col]].copy()
        subset[likert_col] = pd.to_numeric(subset[likert_col], errors="coerce")
        subset = subset.dropna()

        if socio_label == "Experiência":
            subset = subset[subset[socio_col] != "Prefiro não responder"].copy()

        # Filtrar grupos com n < 5
        group_counts = subset[socio_col].value_counts()
        valid_groups = group_counts[group_counts >= 5].index
        subset = subset[subset[socio_col].isin(valid_groups)]

        if subset[socio_col].nunique() < 2:
            continue

        try:
            dunn = sp.posthoc_dunn(
                subset,
                val_col=likert_col,
                group_col=socio_col,
                p_adjust='bonferroni'
            )
        except Exception as e:
            print(f"  ⚠️  Erro no post-hoc {socio_label} × {q_label}: {e}")
            continue

        # Extrair pares significativos
        likert_short = likert_col.split("]")[0].replace("[", "").strip() if "]" in likert_col else likert_col

        print(f"\n  {socio_label} × {q_label} ({likert_short})")
        print(f"    Kruskal-Wallis: H={d['test_stat']:.2f}, p={d['p_value']:.4f}, ε²={d['effect_size']:.3f} (efeito {d['effect_label']})")
        print(f"    Post-hoc Dunn (Bonferroni):")

        found_sig = False
        for g1_idx, g1 in enumerate(dunn.index):
            for g2_idx, g2 in enumerate(dunn.columns):
                if g1_idx >= g2_idx:
                    continue
                p_dunn = dunn.loc[g1, g2]
                if p_dunn < 0.05:
                    found_sig = True
                    stars = "***" if p_dunn < 0.001 else "**" if p_dunn < 0.01 else "*"
                    print(f"      {g1} vs {g2}: p={p_dunn:.4f} {stars}")

                    posthoc_results.append({
                        "socio_label": socio_label,
                        "q_label": q_label,
                        "likert_col": likert_short,
                        "H_stat": d["test_stat"],
                        "kw_p_value": d["p_value"],
                        "epsilon_sq": d["effect_size"],
                        "effect_label": d["effect_label"],
                        "group_1": g1,
                        "group_2": g2,
                        "dunn_p_bonferroni": p_dunn,
                    })

        if not found_sig:
            print(f"      (nenhum par significativo após correção de Bonferroni)")

    return posthoc_results


def save_posthoc_csv(posthoc_results: List[Dict], details: List[Dict],
                     out_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Salva resultados post-hoc e resumo de efeitos em CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    posthoc_path = None
    effects_path = None

    # 1. CSV de post-hoc
    if posthoc_results:
        df_posthoc = pd.DataFrame(posthoc_results)
        posthoc_path = out_dir / "posthoc_dunn_bonferroni.csv"
        df_posthoc.to_csv(posthoc_path, index=False)
        print(f"\n✅ Post-hoc salvo em: {posthoc_path}")

    # 2. CSV de efeitos (todas as combinações significativas)
    sig_details = [d for d in details if d["p_value"] < 0.05]
    if sig_details:
        rows = []
        for d in sig_details:
            row = {
                "socio_label": d["socio_label"],
                "q_label": d["q_label"],
                "test_type": d["test_type"],
                "test_stat": round(d["test_stat"], 4),
                "p_value": round(d["p_value"], 6),
                "effect_size": round(d["effect_size"], 4),
                "effect_label": d["effect_label"],
            }
            rows.append(row)
        df_effects = pd.DataFrame(rows)
        effects_path = out_dir / "effect_sizes.csv"
        df_effects.to_csv(effects_path, index=False)
        print(f"✅ Tamanhos de efeito salvos em: {effects_path}")

    return posthoc_path, effects_path


def plot_heatmap(pvalue_matrix: pd.DataFrame, out_dir: Path,
                 lang: str = "pt") -> Optional[Path]:
    """
    Plota heatmap com p-valores, destaca em negrito valores < 0.05.
    Usa paleta de laranja do projeto.

    lang: "pt" para português, "en" para inglês.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = pvalue_matrix.copy()
    if lang == "en":
        matrix.columns = [SOCIO_LABEL_EN.get(c, c) for c in matrix.columns]
        xlabel = "Sociodemographic Variables"
        ylabel = "Effectiveness and Quality Variables"
        suffix = "_en"
    else:
        xlabel = "Variáveis Sociodemográficas"
        ylabel = "Variáveis Efetividade e Qualidade"
        suffix = ""

    fig, ax = plt.subplots(figsize=(12, 14))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="gray",
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.05,
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for i, q_idx in enumerate(matrix.index):
        for j, socio_col in enumerate(matrix.columns):
            pval = matrix.iloc[i, j]

            if pd.notna(pval):
                if pval < 0.05:
                    if pval < 0.001:
                        color = BASE_ORANGE
                    elif pval < 0.01:
                        color = "#ff8533"
                    else:
                        color = "#ffb366"

                    rect = plt.Rectangle((j, i), 1, 1, fill=True,
                                        facecolor=color, edgecolor="gray",
                                        linewidth=0.5, zorder=2)
                    ax.add_patch(rect)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=True,
                                        facecolor="white", edgecolor="gray",
                                        linewidth=0.5, zorder=2)
                    ax.add_patch(rect)

    for text in ax.texts:
        try:
            val = float(text.get_text())
            if val < 0.05:
                text.set_weight("bold")
                text.set_color("white")
                text.set_fontsize(12)
            else:
                text.set_fontsize(12)
                text.set_color("black")
        except ValueError:
            pass

    plt.tight_layout()
    out_path = out_dir / f"significancia_heatmap{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Heatmap salvo em: {out_path}")
    return out_path


def print_ranking(pvalue_matrix: pd.DataFrame, details: List[Dict]) -> None:
    """Imprime ranking de diferenças significativas (p < 0.05) com tamanho de efeito."""
    print("\n" + "="*80)
    print("📊 RANKING DE SIGNIFICÂNCIA ESTATÍSTICA (p < 0.05)")
    print("="*80)

    sig_details = [d for d in details if d["p_value"] < 0.05]

    if not sig_details:
        print("❌ Nenhuma diferença estatisticamente significativa encontrada (α=0.05)")
        return

    sig_details.sort(key=lambda x: x["p_value"])

    print(f"\n🎯 Total de combinações significativas: {len(sig_details)}\n")

    for rank, d in enumerate(sig_details, 1):
        stars = "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*"
        if d["test_type"] == "spearman":
            stat_str = f"r={d['test_stat']:.3f}"
        else:
            stat_str = f"H={d['test_stat']:.2f}, ε²={d['effect_size']:.3f}"
        print(f"{rank:2d}. {d['q_label']:3s} × {d['socio_label']:18s} | p={d['p_value']:.6f} {stars} | {stat_str} (efeito {d['effect_label']})")

    print("\n" + "="*80)


def analyze_significancia(csv_path: Optional[Path] = None, 
                         out_dir: Optional[Path] = None) -> Dict:
    """
    Pipeline completo: carrega, processa, calcula p-valores e plota heatmap.
    """
    project_root = Path(__file__).resolve().parents[1]
    
    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "tratado.csv"
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output"
    
    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return {"error": f"CSV não encontrado: {csv_path}"}
    
    print(f"📂 Carregando: {csv_path}")
    
    # 1️⃣ Carregamento e pré-processamento
    df, likert_cols, socio_cols_found = load_and_preprocess(csv_path)
    
    if not likert_cols or not socio_cols_found:
        return {
            "error": "Colunas Likert ou sociodemográficas não encontradas",
            "likert_count": len(likert_cols),
            "socio_count": len(socio_cols_found),
        }
    
    # 2️⃣ Preparar dados Likert
    df_likert = prepare_likert_data(df, likert_cols)
    print(f"✓ Dados Likert: {df_likert.shape}")
    
    # Expandir df_likert com colunas sociodemográficas
    for socio_col in socio_cols_found:
        df_likert[socio_col] = df[socio_col]
    
    # 3️⃣ Calcular p-valores e tamanhos de efeito
    pvalue_matrix, details = calculate_pvalues(df_likert, likert_cols, socio_cols_found)
    print(f"✓ Matriz de p-valores: {pvalue_matrix.shape}")

    # 4️⃣ Plotar heatmaps (PT e EN)
    heatmap_path = plot_heatmap(pvalue_matrix, out_dir, lang="pt")
    heatmap_path_en = plot_heatmap(pvalue_matrix, out_dir, lang="en")

    # 5️⃣ Imprimir ranking com tamanhos de efeito
    print_ranking(pvalue_matrix, details)

    # 6️⃣ Post-hoc de Dunn (Bonferroni) para Kruskal-Wallis significativos
    posthoc_results = run_posthoc(df_likert, details, likert_cols)

    # 7️⃣ Salvar CSVs de post-hoc e tamanhos de efeito
    posthoc_path, effects_path = save_posthoc_csv(posthoc_results, details, out_dir)

    return {
        "status": "ok",
        "n_questions": len(likert_cols),
        "n_sociodemographic": len(socio_cols_found),
        "pvalue_matrix_shape": pvalue_matrix.shape,
        "heatmap_path": heatmap_path,
        "heatmap_path_en": heatmap_path_en,
        "pvalue_matrix": pvalue_matrix,
        "details": details,
        "posthoc_results": posthoc_results,
        "posthoc_csv": posthoc_path,
        "effects_csv": effects_path,
    }


if __name__ == "__main__":
    import sys
    
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analyze_significancia(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Questões: {result['n_questions']}")
        print(f"   Sociodemográficos: {result['n_sociodemographic']}")
        print(f"   Heatmap: {result['heatmap_path']}")