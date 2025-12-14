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
EXPERIENCE_MAP_NORM = {
    "até um ano": 0.5,
    "entre 1 e 2 anos": 1.5,
    "entre 3 e 4 anos": 3.5,
    "entre 5 e 6 anos": 5.5,
    "entre 7 e 8 anos": 7.5,
    "mais de 8 anos": 10.0,
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
    
    # Normalizar experiência
    df_clean["exp_norm"] = (
        df_clean[exp_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )
    
    # Mapear experiência para valores numéricos
    df_clean["exp_years"] = df_clean["exp_norm"].map(EXPERIENCE_MAP_NORM)
    print(f"\n[DEBUG] Valores únicos em {exp_col}:")
    print(df_clean[exp_col].value_counts())
    
    df_clean = df_clean.dropna(subset=["exp_years"])
    print(f"[DEBUG] Linhas após mapear experiência: {len(df_clean)}")
    
    # Normalizar nível (remover espaços extras)
    df_clean["nivel"] = df_clean[nivel_col].astype(str).str.strip()
     
    print(f"\n[DEBUG] Valores únicos em {nivel_col}:")
    print(df_clean[nivel_col].value_counts())
    print(f"\n[DEBUG] Dados para plotar:")
    print(df_clean.head(20))
     
    # Criar boxplot
    boxplot_path = None
    if plt is not None and sns is not None:
        try:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # ⭐ Filtrar apenas níveis com n >= 5
            niveis_com_dados = sorted([
                nivel for nivel in df_clean["nivel"].unique()
                if (df_clean["nivel"] == nivel).sum() >= 5
            ])
            
            if len(niveis_com_dados) < 2:
                print(f"❌ Apenas {len(niveis_com_dados)} nível(is) com n >= 5. Análise impossível.")
                return {"error": "Dados insuficientes", "boxplot_path": None}
            
            # Filtrar dataframe
            df_plot = df_clean[df_clean["nivel"].isin(niveis_com_dados)].copy()
            
            # Remover outliers antes de calcular a média
            Q1 = df_plot.groupby("nivel", observed=True)["exp_years"].quantile(0.25)
            Q3 = df_plot.groupby("nivel", observed=True)["exp_years"].quantile(0.75)
            IQR = Q3 - Q1
            
            # Filtrar dados sem outliers
            df_no_outliers = df_plot.copy()
            for nivel in niveis_com_dados:
                if pd.isna(nivel):
                    continue
                mask = (df_no_outliers["nivel"] == nivel)
                lower = Q1[nivel] - 1.5 * IQR[nivel]
                upper = Q3[nivel] + 1.5 * IQR[nivel]
                df_no_outliers.loc[mask, "exp_years"] = df_no_outliers.loc[mask, "exp_years"].clip(lower, upper)
             
            # Mapear cores para que a maior média fique na cor base (#ff6002)
            means_for_color = df_no_outliers.groupby("nivel", observed=True)["exp_years"].mean()
            sorted_niveis = means_for_color.sort_values().index.tolist()
            
            # Gerar paleta de cores laranja
            orange_palette = _orange_shades(BASE_ORANGE, len(sorted_niveis))
            color_map = {nivel: orange_palette[i] for i, nivel in enumerate(sorted_niveis)}
            
            # Boxplot com paleta laranja
            sns.boxplot(
                data=df_plot,
                x="nivel",
                y="exp_years",
                order=sorted_niveis,
                ax=ax,
                palette=color_map,
                width=0.7,
                linewidth=2,
                showfliers=False,
            )
            
            # Calcular média SEM outliers
            means = df_no_outliers.groupby("nivel", observed=True)["exp_years"].mean()
            positions = range(len(means))
            
            # Plotar médias como círculo preto com borda branca (PADRÃO DO PROJETO)
            means_ordered = [means[nivel] for nivel in sorted_niveis]
            
            ax.scatter(
                positions,
                means_ordered,
                color="black",          # preenchimento preto
                s=150,
                zorder=5,
                marker="o",
                edgecolor="white",      # borda branca
                linewidth=2.5,          # espessura da borda
                label="Média"
            )
            
            # Valores numéricos das médias
            for pos, nivel in enumerate(sorted_niveis):
                mean_val = means[nivel]
                ax.text(
                    pos,
                    mean_val + 0.3,
                    f"{mean_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    fontweight="bold",
                    color="black"
                )
            
            # Criar rótulos com n
            group_labels = [
                f"{nivel.replace(' (', '\n(')}\n"
                f"(n={(df_plot['nivel'] == nivel).sum()})"
                for nivel in sorted_niveis
            ]
            
            ax.set_xlabel("Nível Profissional", fontsize=20, fontweight="bold")
            ax.set_ylabel("Tempo de Experiência (anos)", fontsize=20, fontweight="bold")
            ax.set_xticklabels(group_labels, ha="center", fontsize=16)
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
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "tratado.csv"))
    
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
