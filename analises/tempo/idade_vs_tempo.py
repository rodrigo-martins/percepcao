from pathlib import Path
from typing import Optional, Dict, Tuple
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

# Mapeamento da escala Likert
LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}


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


def load_and_prepare(csv_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Carrega CSV, mapeia Q6 e Idade para numérico.
    Retorna: (df_clean, stats_dict)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None
    
    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Encontrar coluna Q6 (carga horária) e Idade
    q6_col = None
    age_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Procurar por Q6 (carga horária) - começa com '['
        if col_lower.startswith("[") and "carga horária" in col_lower and "prejudicou" in col_lower:
            q6_col = col
        
        # Procurar por "Qual é a sua idade?" - NÃO começa com '['
        if col_lower == "qual é a sua idade?" or col_lower == "qual e a sua idade?":
            age_col = col
    
    if q6_col is None or age_col is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Q6 (carga horária): {q6_col}")
        print(f"   Idade: {age_col}")
        print(f"\nColunas disponíveis:")
        for i, col in enumerate(df.columns):
            print(f"   {i}: {repr(col)}")
        return None, None
    
    print(f"✓ Q6 coluna: {repr(q6_col)}")
    print(f"✓ Idade coluna: {repr(age_col)}")
    
    # Preparar dados
    df_clean = df[[age_col, q6_col]].copy()
    df_clean.columns = ["idade", "q6"]
    
    # Mapear Q6 para numérico
    df_clean["q6"] = df_clean["q6"].map(LIKERT_MAP)
    
    # Converter Idade para numérico
    df_clean["idade"] = df_clean["idade"].apply(convert_age_to_numeric)
    
    # DEBUG: verificar dados após conversão
    print(f"\n[DEBUG] Amostra de dados após conversão:")
    print(df_clean.head(10))
    print(f"[DEBUG] Valores não-nulos: idade={df_clean['idade'].notna().sum()}, q6={df_clean['q6'].notna().sum()}")
    
    # Remover NaNs
    df_clean = df_clean.dropna()
    
    print(f"✓ Dados limpos: {len(df_clean)} linhas válidas")
    
    if len(df_clean) < 10:
        print(f"❌ Dados insuficientes para análise (n={len(df_clean)})")
        return None, None
    
    # Calcular correlação de Spearman
    r, pval = stats.spearmanr(df_clean["idade"], df_clean["q6"])
    
    stats_dict = {
        "r": r,
        "pval": pval,
        "n": len(df_clean),
    }
    
    print(f"\n📊 Correlação de Spearman:")
    print(f"   r = {r:.4f}")
    print(f"   p-valor = {pval:.6f}")
    print(f"   n = {len(df_clean)}")
    
    return df_clean, stats_dict


def plot_scatter_regression(df_clean: pd.DataFrame, stats_dict: Dict, 
                            out_dir: Path) -> Optional[Path]:
    """
    Cria scatter plot com linha de regressão, estilo laranja.
    Inclui quadro com teste estatístico de correlação.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot com linha de regressão
    sns.regplot(
        data=df_clean,
        x="idade",
        y="q6",
        scatter_kws={
            "s": 80,
            "alpha": 0.6,
            "color": BASE_ORANGE,
            "edgecolor": "white",
            "linewidths": 0.5,
        },
        line_kws={
            "color": BASE_ORANGE,
            "linewidth": 3,
        },
        ax=ax,
    )
    
    # Títulos e labels
    ax.set_title(
        "Correlação: Idade vs. Q6 (Carga horária prejudicou tempo pessoal)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Idade (anos)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Q6 (Concordância 1-5)", fontsize=20, fontweight="bold")
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis='both', labelsize=18)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Anotação com estatísticas
    r = stats_dict["r"]
    pval = stats_dict["pval"]
    n = stats_dict["n"]
    
    # Determinar significância
    if pval < 0.001:
        sig_symbol = "***"
        sig_text = "Altamente significativo"
    elif pval < 0.01:
        sig_symbol = "**"
        sig_text = "Muito significativo"
    elif pval < 0.05:
        sig_symbol = "*"
        sig_text = "Significativo"
    else:
        sig_symbol = "ns"
        sig_text = "Não significativo"
    
    # Criar quadro conciso de estatísticas
    quadro_text = (
        f"Correlação de Spearman\n"
        f"{'─' * 32}\n"
        f"r = {r:.4f}\n"
        f"p = {pval:.6f}\n"
        f"n = {n}\n"
        f"Resultado: {sig_symbol} {sig_text}"
    )
    
    ax.text(
        0.05,
        0.95,
        quadro_text,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="black", linewidth=2),
        color="black",
    )
    
    plt.tight_layout()
    
    # Salvar figura
    out_path = out_dir / "scatter_idade_q6_orange.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"\n✅ Scatter plot salvo em: {out_path}")
    return out_path


def analyze_idade_vs_tempo(csv_path: Optional[Path] = None,
                           out_dir: Optional[Path] = None) -> Dict:
    """
    Pipeline completo: carrega dados, calcula correlação e gera scatter plot.
    """
    project_root = Path(__file__).resolve().parents[2]  # CORRIGIDO: .parents[1] → .parents[2]
    
    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "ordered.csv"
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output"
    
    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return {"error": f"CSV não encontrado: {csv_path}"}
    
    print(f"📂 Carregando: {csv_path}\n")
    
    # 1️⃣ Carregar e preparar dados
    df_clean, stats_dict = load_and_prepare(csv_path)
    
    if df_clean is None or stats_dict is None:
        return {"error": "Erro ao preparar dados"}
    
    # 2️⃣ Plotar scatter com regressão
    plot_path = plot_scatter_regression(df_clean, stats_dict, out_dir)
    
    return {
        "status": "ok",
        "r": stats_dict["r"],
        "pval": stats_dict["pval"],
        "n": stats_dict["n"],
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    import sys
    
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analyze_idade_vs_tempo(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   r (Spearman): {result['r']:.4f}")
        print(f"   p-valor: {result['pval']:.6f}")
        print(f"   n: {result['n']}")
        print(f"   Plot: {result['plot_path']}")