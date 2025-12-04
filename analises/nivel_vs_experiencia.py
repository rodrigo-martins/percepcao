from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
except Exception:
    plt = None
    sns = None
    mcolors = None
    np = None

BASE_ORANGE = "#ff6002"

# Mapeamentos de experiência para valores numéricos (ponto médio em anos)
EXPERIENCE_MAP = {
    "Até um ano.": 0.5,
    "Entre 1 e 2 anos.": 1.5,
    "Entre 3 e 4 anos.": 3.5,
    "Entre 5 e 6 anos.": 5.5,
    "Entre 7 e 8 anos.": 7.5,
    "Mais de 8 anos.": 10.0,
}

# Ordem dos níveis profissionais
NIVEL_ORDER = [
    "Estudante/ Estagiário / Trainee",
    "Júnior",
    "Pleno",
    "Sênior",
    "Especialista / Principal (Foco em contribuição técnica individual)",
    "Líder / Coordenador / Gerente (Foco em gestão de pessoas)",
]

# Mapa de labels simplificados para exibição
NIVEL_LABELS_MAP = {
    "Estudante/ Estagiário / Trainee": "Estagiário",
    "Júnior": "Júnior",
    "Pleno": "Pleno",
    "Sênior": "Sênior",
    "Especialista / Principal (Foco em contribuição técnica individual)": "Especialista (técnico)",
    "Líder / Coordenador / Gerente (Foco em gestão de pessoas)": "Líder (pessoas)",
}


def _orange_shades(base_hex: str, n: int) -> List[tuple]:
    """Gera n tons do laranja base: menor -> mais claro, maior -> mais escuro."""
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


def _normalize_text(s: str) -> str:
    """Normalizar texto removendo espaços extras e convertendo para lowercase."""
    if s is None:
        return ""
    return str(s).strip().lower()


def analyze_nivel_vs_experiencia(
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analisa nível profissional vs tempo de experiência.
    Cria boxplot separado para cada nível profissional com tons de laranja.
    """
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detectar colunas exatas
    print("\n[DEBUG] Colunas disponíveis no CSV:")
    for col in df.columns:
        print(f"  - {repr(col)}")
    
    # Encontrar coluna de nível profissional
    nivel_col = None
    for col in df.columns:
        if "nível" in col.lower() or "nivel" in col.lower():
            if "profissional" in col.lower():
                nivel_col = col
                break
    
    # Encontrar coluna de experiência
    exp_col = None
    for col in df.columns:
        if "experiência" in col.lower() or "experiencia" in col.lower():
            if "tempo" in col.lower() or "total" in col.lower():
                exp_col = col
                break
    
    print(f"\n[DEBUG] Coluna nível encontrada: {repr(nivel_col)}")
    print(f"[DEBUG] Coluna experiência encontrada: {repr(exp_col)}")
    
    if nivel_col is None or exp_col is None:
        print(f"[ERRO] Colunas não encontradas!")
        return {"error": "Colunas não encontradas", "boxplot_path": None}
    
    # Filtrar dados válidos
    df_clean = df[[nivel_col, exp_col]].copy()
    print(f"\n[DEBUG] Linhas antes de dropna: {len(df_clean)}")
    
    df_clean = df_clean.dropna()
    print(f"[DEBUG] Linhas após dropna: {len(df_clean)}")
    
    # Mapear experiência para valores numéricos
    df_clean["exp_years"] = df_clean[exp_col].map(EXPERIENCE_MAP)
    print(f"\n[DEBUG] Valores únicos em {exp_col}:")
    print(df_clean[exp_col].value_counts())
    
    df_clean = df_clean.dropna(subset=["exp_years"])
    print(f"[DEBUG] Linhas após mapear experiência: {len(df_clean)}")
    
    # Garantir que nível está na ordem desejada
    df_clean[nivel_col] = pd.Categorical(
        df_clean[nivel_col],
        categories=NIVEL_ORDER,
        ordered=True
    )
    
    # Mapear labels simplificados para exibição
    df_clean["nivel_display"] = df_clean[nivel_col].map(NIVEL_LABELS_MAP)
    
    # Remover linhas com nivel_display = NaN
    df_clean = df_clean.dropna(subset=["nivel_display"])
     
    print(f"\n[DEBUG] Valores únicos em {nivel_col}:")
    print(df_clean[nivel_col].value_counts())
    print(f"\n[DEBUG] Dados para plotar:")
    print(df_clean.head(20))
     
    # Criar boxplot
    boxplot_path = None
    if plt is not None and sns is not None:
        try:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Remover outliers antes de calcular a média
            Q1 = df_clean.groupby("nivel_display", observed=True)["exp_years"].quantile(0.25)
            Q3 = df_clean.groupby("nivel_display", observed=True)["exp_years"].quantile(0.75)
            IQR = Q3 - Q1
            
            # Filtrar dados sem outliers
            df_no_outliers = df_clean.copy()
            for nivel in df_no_outliers["nivel_display"].unique():
                if pd.isna(nivel):
                    continue
                mask = (df_no_outliers["nivel_display"] == nivel)
                lower = Q1[nivel] - 1.5 * IQR[nivel]
                upper = Q3[nivel] + 1.5 * IQR[nivel]
                df_no_outliers.loc[mask, "exp_years"] = df_no_outliers.loc[mask, "exp_years"].clip(lower, upper)
             
            # Gerar paleta de cores laranja
            n_niveis = len(NIVEL_LABELS_MAP)
            orange_palette = _orange_shades(BASE_ORANGE, n_niveis)
            
            # Mapear cores para que a maior média fique na cor base (#ff6002)
            means_for_color = df_no_outliers.groupby("nivel_display", observed=True)["exp_years"].mean()
            sorted_niveis = means_for_color.sort_values(ascending=True).index.tolist()
            color_map = {nivel: orange_palette[i] for i, nivel in enumerate(sorted_niveis)}
            
            # Boxplot com paleta laranja
            sns.boxplot(
                data=df_clean,
                x="nivel_display",
                y="exp_years",
                ax=ax,
                palette=color_map,
                width=0.7,
                linewidth=2,
                showfliers=False,
            )
            
            # Calcular média SEM outliers
            means = df_no_outliers.groupby("nivel_display", observed=True)["exp_years"].mean()
            positions = range(len(means))
            
            # Valores da média acima da linha
            for pos, mean_val in zip(positions, means.values):
                if pos == 0:
                    ax.hlines(mean_val, pos - 0.35, pos + 0.35, color="black", linestyles="--", 
                             linewidth=1.5, zorder=4, label="Média (sem outliers)")
                else:
                    ax.hlines(mean_val, pos - 0.35, pos + 0.35, color="black", linestyles="--", 
                             linewidth=1.5, zorder=4)
                 
                # Valor numérico acima
                ax.text(pos, mean_val + 0.3, f"{mean_val:.1f}", ha="center", va="center",
                        fontsize=20, fontweight="bold", color="black")
            
            ax.set_xlabel("Nível Profissional", fontsize=20, fontweight="bold")
            ax.set_ylabel("Tempo de Experiência (anos)", fontsize=20, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), ha="center", fontsize=20, rotation=15)
            ax.tick_params(axis="y", labelsize=20)
            ax.grid(axis="y", alpha=0.35, linestyle="--")
            ax.legend(loc="upper left", fontsize=20)
            
            plt.tight_layout()
            boxplot_path = out_dir / "nivel_vs_experiencia_boxplot.png"
            fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            print(f"\n✅ Boxplot salvo em: {boxplot_path}\n")
        except Exception as e:
            print(f"\n❌ Erro ao criar boxplot: {e}\n")
            import traceback
            traceback.print_exc()
    
    return {
        "error": None,
        "boxplot_path": boxplot_path,
    }


if __name__ == "__main__":
    import sys
    import os
    
    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    
    p = Path(csv_path)
    if not p.exists():
        print(f"CSV não encontrado: {p}")
        raise SystemExit(1)
    
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_nivel_vs_experiencia(df)
    
    if res.get("error"):
        print(f"Erro: {res['error']}")
    else:
        print(f"✅ Análise concluída!")
        print(f"📊 Boxplot: {res['boxplot_path']}")
