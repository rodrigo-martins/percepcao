from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import sys
import re

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

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    TUKEY_AVAILABLE = True
except ImportError:
    pairwise_tukeyhsd = None
    TUKEY_AVAILABLE = False

BASE_ORANGE = "#ff6002"

# Mapeamento da escala Likert
LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

# Mapeamento de Estados para Regiões
REGION_MAP = {
    'Acre': 'Norte', 'Amapá': 'Norte', 'Amazonas': 'Norte', 'Pará': 'Norte', 
    'Rondônia': 'Norte', 'Roraima': 'Norte', 'Tocantins': 'Norte',
    'Alagoas': 'Nordeste', 'Bahia': 'Nordeste', 'Ceará': 'Nordeste', 
    'Maranhão': 'Nordeste', 'Paraíba': 'Nordeste', 'Pernambuco': 'Nordeste', 
    'Piauí': 'Nordeste', 'Rio Grande do Norte': 'Nordeste', 'Sergipe': 'Nordeste',
    'Distrito Federal': 'Centro-Oeste', 'Goiás': 'Centro-Oeste', 
    'Mato Grosso': 'Centro-Oeste', 'Mato Grosso do Sul': 'Centro-Oeste',
    'Espírito Santo': 'Sudeste', 'Minas Gerais': 'Sudeste', 
    'Rio de Janeiro': 'Sudeste', 'São Paulo': 'Sudeste',
    'Paraná': 'Sul', 'Rio Grande do Sul': 'Sul', 'Santa Catarina': 'Sul'
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
        t = i / max(1, n - 1)  # 0 = base (escuro), 1 = branco (claro)
        mix = 0.35 + 0.65 * t
        rgb = white * (1.0 - mix) + base_rgb * mix
        shades.append(tuple(rgb.tolist()))
    return shades


def clean_estado(estado_str: str) -> Optional[str]:
    """Remove sufixos como '(UF)' da string de estado."""
    if pd.isna(estado_str):
        return None
    
    estado = str(estado_str).strip()
    
    # Remover sufixos com UF entre parênteses, ex: "(PE)", "(SP)"
    estado = re.sub(r'\s*\([A-Z]{2}\)\s*$', '', estado).strip()
    
    return estado if estado else None


def load_and_prepare(csv_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Carrega CSV, mapeia Q10 e Estado para numérico/string, cria coluna Região.
    Retorna: (df_clean, stats_dict)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None
    
    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Encontrar coluna Q10 (tempo suficiente) e Estado
    q10_col = None
    estado_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Procurar por Q10 (tempo suficiente)
        if col_lower.startswith("[") and "tempo suficiente" in col_lower:
            q10_col = col
        
        # Procurar por Estado - ser mais flexível
        if "estado" in col_lower:
            estado_col = col
    
    if q10_col is None or estado_col is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Q10 (tempo suficiente): {q10_col}")
        print(f"   Estado: {estado_col}")
        print(f"\n🔍 Colunas disponíveis (primeiras 20):")
        for i, col in enumerate(df.columns[:20]):
            col_lower = str(col).lower()
            starts_bracket = "[" if str(col).strip().startswith("[") else " "
            print(f"   {starts_bracket} {i:2d}: {repr(col)}")
        return None, None
    
    print(f"✓ Q10 coluna: {repr(q10_col)}")
    print(f"✓ Estado coluna: {repr(estado_col)}")
    
    # Preparar dados
    df_clean = df[[estado_col, q10_col]].copy()
    df_clean.columns = ["estado", "q10"]
    
    print(f"\n[DEBUG] Primeiras 5 valores de estado:")
    print(df_clean["estado"].head())
    print(f"\n[DEBUG] Primeiras 5 valores de q10:")
    print(df_clean["q10"].head())
    
    # Mapear Q10 para numérico
    df_clean["q10"] = df_clean["q10"].map(LIKERT_MAP)
    df_clean["q10"] = pd.to_numeric(df_clean["q10"], errors="coerce")
    
    # Limpar Estado
    df_clean["estado"] = df_clean["estado"].apply(clean_estado)
    
    print(f"\n[DEBUG] Primeiras 5 valores de estado após limpeza:")
    print(df_clean["estado"].head())
    
    # Remover NaNs
    df_clean = df_clean.dropna(subset=["estado", "q10"])
    
    print(f"✓ Dados limpos: {len(df_clean)} linhas válidas")
    
    if len(df_clean) < 10:
        print(f"❌ Dados insuficientes para análise (n={len(df_clean)})")
        return None, None
    
    # Debug: mostrar estados únicos encontrados
    print(f"\n[DEBUG] Estados encontrados:")
    print(df_clean["estado"].unique())
    
    # Mapear Estado para Região
    df_clean["Regiao"] = df_clean["estado"].map(REGION_MAP)
    
    # Remover estados não mapeados
    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["Regiao"])
    n_after = len(df_clean)
    
    print(f"✓ Regiões mapeadas: {n_after} linhas válidas ({n_before - n_after} não mapeados)")
    
    if n_after == 0:
        print(f"❌ Nenhum estado foi mapeado para região!")
        print(f"Estados não mapeados:")
        print(df_clean[df_clean["Regiao"].isna()]["estado"].unique())
        return None, None
    
    print(f"\n📊 Distribuição por Região:")
    for regiao in sorted(df_clean["Regiao"].unique()):
        count = (df_clean["Regiao"] == regiao).sum()
        mean = df_clean[df_clean["Regiao"] == regiao]["q10"].mean()
        print(f"   {regiao}: {count} respondentes, média = {mean:.2f}")
    
    return df_clean, {"n_regions": df_clean["Regiao"].nunique()}


def plot_boxplot_regions(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot com paleta laranja mostrando distribuição de Q10 por região.
    Inclui quadro com estatísticas e significância.
    Regiões ordenadas da menor para maior média.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcular médias por região e ordenar (menor → maior)
    region_means = df_clean.groupby("Regiao")["q10"].mean().sort_values(ascending=True)
    region_order = region_means.index.tolist()
    
    print(f"\n📈 Ordem das regiões (menor → maior média):")
    for i, regiao in enumerate(region_order, 1):
        print(f"   {i}. {regiao}: {region_means[regiao]:.2f}")
    
    # Gerar paleta de laranja (regiões em ordem)
    n_regions = len(region_order)
    orange_palette = _orange_shades(BASE_ORANGE, n_regions)
    
    # Criar dicionário de cores baseado na ordem
    color_dict = {regiao: orange_palette[i] for i, regiao in enumerate(region_order)}
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plotar boxplot
    bp = sns.boxplot(
        data=df_clean,
        x="Regiao",
        y="q10",
        order=region_order,
        hue="Regiao",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    means = df_clean.groupby("Regiao", observed=True)["q10"].mean()
    positions = range(len(region_order))
    
    means_ordered = [means[regiao] for regiao in region_order]
    
    ax.scatter(
        positions,
        means_ordered,
        color="black",
        s=150,
        zorder=3,
        marker="o",
        edgecolor="white",
        linewidth=2,
        label="Média"
    )
    
    # Adicionar valores das médias como texto
    for pos, regiao in enumerate(region_order):
        mean_val = means_ordered[pos]
        
        text = ax.text(
            pos,
            mean_val + 0.15,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color="black"
        )
        
        # Adicionar borda branca
        if path_effects is not None:
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ])
    
    # Criar rótulos com n de cada região
    region_labels = []
    for regiao in region_order:
        n_respondentes = (df_clean["Regiao"] == regiao).sum()
        label = f"{regiao}\n(n={n_respondentes})"
        region_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel("Região do Brasil", fontsize=20, fontweight="bold")
    ax.set_xticklabels(region_labels, fontsize=16)
    ax.set_ylabel("Q10 (Tempo no expediente)", fontsize=20, fontweight="bold")
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis="both", labelsize=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)
    
    # Legenda
    ax.legend(loc="upper right", fontsize=20)
    
    # Calcular estatísticas para o quadro
    grupos = {}
    for regiao in df_clean["Regiao"].unique():
        subset = df_clean[df_clean["Regiao"] == regiao]["q10"].values
        if len(subset) >= 5:
            grupos[regiao] = subset
    
    if len(grupos) >= 2:
        # Teste de Kruskal-Wallis
        h_stat, p_valor = stats.kruskal(*grupos.values())
        
        # Determinar significância
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
        
        # Criar quadro conciso de estatísticas
        quadro_text = (
            f"Teste de Kruskal-Wallis\n"
            f"{'─' * 32}\n"
            f"n = {sum(len(g) for g in grupos.values())}\n"
            f"H = {h_stat:.4f}\n"
            f"p = {p_valor:.6f}\n"
            f"Resultado: {sig_symbol} {sig_text}"
        )
        
        # Adicionar quadro ao gráfico
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(
            0.98, 0.02,
            quadro_text,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontfamily='monospace',
            bbox=props
        )
    
    plt.tight_layout()
    
    # Salvar figura
    script_name = Path(__file__).stem
    out_path = out_dir / f"{script_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"\n✅ Boxplot salvo em: {out_path}")
    return out_path


def analise_competitividade_regiao(csv_path: Optional[Path] = None,
                                    out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, mapeia regiões, plota boxplot e realiza teste de Kruskal-Wallis para Q10.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "tratado.csv"
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output"
    
    if not csv_path.exists():
        print(f"❌ CSV não encontrado: {csv_path}")
        return {"error": f"CSV não encontrado: {csv_path}"}
    
    print(f"📂 Carregando: {csv_path}\n")
    
    # Preparar dados
    df_clean, prep_info = load_and_prepare(csv_path)
    
    if df_clean is None:
        return {"error": "Falha ao preparar dados"}
    
    # Plotar boxplot
    boxplot_path = plot_boxplot_regions(df_clean, out_dir)
    
    # Estatísticas descritivas
    print(f"\n📊 Estatísticas por Região:")
    for regiao in sorted(df_clean["Regiao"].unique()):
        subset = df_clean[df_clean["Regiao"] == regiao]["q10"]
        if len(subset) > 0:
            print(f"\n  {regiao}:")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
            print(f"    min = {subset.min():.0f}, max = {subset.max():.0f}")
    
    # Teste de Kruskal-Wallis
    print(f"\n\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q10 são iguais entre todas as regiões")
    print(f"Hipótese Alternativa (H1): Pelo menos uma região tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos por região
    grupos = {}
    for regiao in df_clean["Regiao"].unique():
        subset = df_clean[df_clean["Regiao"] == regiao]["q10"].values
        if len(subset) >= 5:  # Filtro: mínimo 5 observações
            grupos[regiao] = subset
    
    if len(grupos) < 2:
        print(f"❌ Dados insuficientes para teste (menos de 2 grupos com n≥5)")
    else:
        # Executar teste de Kruskal-Wallis
        h_stat, p_valor = stats.kruskal(*grupos.values())
        
        print(f"Número de regiões (grupos): {len(grupos)}")
        print(f"Total de observações: {sum(len(v) for v in grupos.values())}\n")
        
        print(f"Estatística H (Kruskal-Wallis): {h_stat:.4f}")
        print(f"P-valor: {p_valor:.6f}")
        print(f"Nível de significância (α): 0.05\n")
        
        # Interpretação
        if p_valor < 0.001:
            sig_level = "***  (Altamente significativo)"
            sig_symbol = "***"
        elif p_valor < 0.01:
            sig_level = "**   (Muito significativo)"
            sig_symbol = "**"
        elif p_valor < 0.05:
            sig_level = "*    (Significativo)"
            sig_symbol = "*"
        else:
            sig_level = "ns   (Não significativo)"
            sig_symbol = "ns"
        
        print(f"Resultado: {sig_level}")
        
        # Criar quadro de significância
        print(f"\n\n📊 QUADRO DE TESTE ESTATÍSTICO")
        print(f"{'=' * 80}")
        test_summary = pd.DataFrame({
            "Teste Estatístico": ["Kruskal-Wallis"],
            "Estatística H": [f"{h_stat:.4f}"],
            "P-valor": [f"{p_valor:.6f}"],
            "α (significância)": ["0.05"],
            "Resultado": [sig_symbol],
            "Interpretação": [sig_level.split("(")[0].strip()]
        })
        print(test_summary.to_string(index=False))
        print(f"{'=' * 80}\n")
        
        # Legenda de significância
        print(f"📌 LEGENDA DE SIGNIFICÂNCIA:")
        print(f"   ns  = Não significativo (p ≥ 0.05)")
        print(f"   *   = Significativo (p < 0.05)")
        print(f"   **  = Muito significativo (p < 0.01)")
        print(f"   *** = Altamente significativo (p < 0.001)\n")
        
        if p_valor < 0.05:
            print(f"\n✅ CONCLUSÃO: Rejeita-se H0")
            print(f"   → Existe diferença significativa na concordância (Q10) entre as regiões")
            print(f"   → As regiões diferem em relação ao grau de concordância sobre")
            print(f"     ter tempo suficiente durante o expediente")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa na concordância (Q10) entre as regiões")
            print(f"   → As regiões têm distribuições semelhantes de resposta")
        
        # Salvar resultados em arquivo
        results_path = out_dir / "kruskal_wallis_regiao_Q10.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("TESTE DE KRUSKAL-WALLIS - Q10 por Região\n")
            f.write("=" * 80 + "\n\n")
            f.write("Hipótese Nula (H0): As distribuições de Q10 são iguais entre todas as regiões\n")
            f.write("Hipótese Alternativa (H1): Pelo menos uma região tem distribuição diferente\n\n")
            f.write(f"Estatística H: {h_stat:.4f}\n")
            f.write(f"P-valor: {p_valor:.6f}\n")
            f.write(f"Nível de significância (α): 0.05\n")
            f.write(f"Número de regiões: {len(grupos)}\n")
            f.write(f"Total de observações: {sum(len(v) for v in grupos.values())}\n\n")
            f.write(f"Resultado: {sig_symbol}\n")
            f.write(f"Interpretação: {sig_level}\n\n")
            
            if p_valor < 0.05:
                f.write("CONCLUSÃO: Rejeita-se H0\n")
                f.write("→ Existe diferença significativa na concordância (Q10) entre as regiões\n")
            else:
                f.write("CONCLUSÃO: Falha em rejeitar H0\n")
                f.write("→ Não há diferença significativa na concordância (Q10) entre as regiões\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("ESTATÍSTICAS DESCRITIVAS POR REGIÃO\n")
            f.write("=" * 80 + "\n\n")
            
            for regiao in sorted(grupos.keys()):
                subset = grupos[regiao]
                f.write(f"{regiao}:\n")
                f.write(f"  n = {len(subset)}\n")
                f.write(f"  Média = {subset.mean():.2f}\n")
                f.write(f"  Mediana = {np.median(subset):.2f}\n")
                f.write(f"  Std = {subset.std():.2f}\n")
                f.write(f"  Min = {subset.min():.0f}, Max = {subset.max():.0f}\n\n")
        
        print(f"\n📄 Resultados salvos em: {results_path}")
    
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_regions": prep_info["n_regions"],
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_competitividade_regiao(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Regiões: {result['n_regions']}")
        print(f"   Boxplot: {result['boxplot_path']}")