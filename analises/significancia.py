from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

BASE_ORANGE = "#ff6002"

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


def calculate_pvalues(df: pd.DataFrame, likert_cols: List[str], 
                     socio_cols_found: List[str]) -> pd.DataFrame:
    """
    Calcula matriz de p-valores entre demográficos e questões Likert.
    
    Regras:
    - Idade: Spearman correlation
    - Outros categóricos: Kruskal-Wallis
    - Ignora grupos com n < 5
    - Remove "Prefiro não responder" de Experiência
    """
    n_questions = len(likert_cols)
    n_socio = len(socio_cols_found)
    
    # Criar matriz vazia
    pvalue_matrix = pd.DataFrame(
        np.nan,
        index=[f"Q{i+1}" for i in range(n_questions)],
        columns=[SOCIODEMOGRAPHIC_COLS[col] for col in socio_cols_found]
    )
    
    print("\n🔬 Calculando p-valores...")
    
    for j, socio_col in enumerate(socio_cols_found):
        socio_label = SOCIODEMOGRAPHIC_COLS[socio_col]
        
        for i, likert_col in enumerate(likert_cols):
            subset = df[[socio_col, likert_col]].copy()
            subset[likert_col] = pd.to_numeric(subset[likert_col], errors="coerce")
            subset = subset.dropna()
            
            # ⭐ REMOVER "Prefiro não responder" de Experiência
            if socio_label == "Experiência":
                subset = subset[subset[socio_col] != "Prefiro não responder"].copy()
            
            if len(subset) < 5:
                continue
            
            pval = None
            
            # Regra 1: Idade → Spearman
            if socio_label == "Idade":
                try:
                    subset["age_numeric"] = subset[socio_col].apply(convert_age_to_numeric)
                    subset = subset.dropna(subset=["age_numeric"])
                    if len(subset) >= 5:
                        stat, pval = stats.spearmanr(subset["age_numeric"], subset[likert_col])
                except Exception:
                    pass
            
            # Regra 2: Categóricos → Kruskal-Wallis
            else:
                try:
                    groups = {}
                    for cat in subset[socio_col].unique():
                        vals = subset[subset[socio_col] == cat][likert_col].values
                        if len(vals) >= 5:  # ← Filtrar grupos pequenos
                            groups[cat] = vals
                    
                    if len(groups) >= 2:
                        stat, pval = stats.kruskal(*groups.values())
                except Exception:
                    pass
            
            if pval is not None:
                pvalue_matrix.iloc[i, j] = pval
                if pval < 0.05:
                    print(f"  ⭐ {socio_label:15} × Q{i+1:2d}: p={pval:.4f} ***")
    
    return pvalue_matrix


def plot_heatmap(pvalue_matrix: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Plota heatmap com p-valores, destaca em negrito valores < 0.05.
    Usa paleta de laranja do projeto.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Usar colormap branco (sem cores)
    cmap = "gray"
     
    # Plotar heatmap
    sns.heatmap(
        pvalue_matrix,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=0.05,  # Apenas valores até 0.05 recebem cor
    )
    
    # Remover título
    # Rótulos do eixo X no topo
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    
    ax.set_xlabel("Variáveis Sociodemográficas", fontsize=14, fontweight="bold")
    ax.set_ylabel("Variáveis Efetividade e Qualidade", fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # ⭐ Colorir apenas células significativas com laranja
    for i, q_idx in enumerate(pvalue_matrix.index):
        for j, socio_col in enumerate(pvalue_matrix.columns):
            pval = pvalue_matrix.iloc[i, j]
            
            if pd.notna(pval):
                if pval < 0.05:
                    # Cores mais escuras para p-valores mais baixos
                    if pval < 0.001:
                        color = BASE_ORANGE  # #ff6002 (escuro)
                    elif pval < 0.01:
                        color = "#ff8533"  # laranja médio
                    else:
                        color = "#ffb366"  # laranja claro
                    
                    # Pintar célula
                    rect = plt.Rectangle((j, i), 1, 1, fill=True, 
                                        facecolor=color, edgecolor="gray", 
                                        linewidth=0.5, zorder=2)
                    ax.add_patch(rect)
                else:
                    # Células não-significativas em branco
                    rect = plt.Rectangle((j, i), 1, 1, fill=True, 
                                        facecolor="white", edgecolor="gray", 
                                        linewidth=0.5, zorder=2)
                    ax.add_patch(rect)
    
    # ⭐ CRÍTICO: Destacar em negrito valores p < 0.05
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
    out_path = out_dir / "significancia_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n✅ Heatmap salvo em: {out_path}")
    return out_path


def print_ranking(pvalue_matrix: pd.DataFrame) -> None:
    """Imprime ranking de diferenças significativas (p < 0.05)."""
    print("\n" + "="*70)
    print("📊 RANKING DE SIGNIFICÂNCIA ESTATÍSTICA (p < 0.05)")
    print("="*70)
    
    significant = []
    for q_idx, row in pvalue_matrix.iterrows():
        for socio_col, pval in row.items():
            if pd.notna(pval) and pval < 0.05:
                significant.append((pval, q_idx, socio_col))
    
    if not significant:
        print("❌ Nenhuma diferença estatisticamente significativa encontrada (α=0.05)")
        return
    
    # Ordenar por p-valor
    significant.sort(key=lambda x: x[0])
    
    print(f"\n🎯 Total de combinações significativas: {len(significant)}\n")
    
    for rank, (pval, q_idx, socio_col) in enumerate(significant, 1):
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
        print(f"{rank:2d}. {q_idx:3s} × {socio_col:18s} | p={pval:.6f} {stars}")
    
    print("\n" + "="*70)


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
    
    # 3️⃣ Calcular p-valores
    pvalue_matrix = calculate_pvalues(df_likert, likert_cols, socio_cols_found)
    print(f"✓ Matriz de p-valores: {pvalue_matrix.shape}")
    
    # 4️⃣ Plotar heatmap
    heatmap_path = plot_heatmap(pvalue_matrix, out_dir)
    
    # 5️⃣ Imprimir ranking
    print_ranking(pvalue_matrix)
    
    return {
        "status": "ok",
        "n_questions": len(likert_cols),
        "n_sociodemographic": len(socio_cols_found),
        "pvalue_matrix_shape": pvalue_matrix.shape,
        "heatmap_path": heatmap_path,
        "pvalue_matrix": pvalue_matrix,
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