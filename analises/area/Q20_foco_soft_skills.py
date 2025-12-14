from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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
    np = None
    path_effects = None

from scipy import stats

BASE_ORANGE = "#ff6002"

# Mapeamento da escala Likert
LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

# Ordem fixa de áreas de atuação (ajustar conforme dados reais)
AREA_ORDER = []


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


def find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    """Busca coluna por palavra-chave (case-insensitive)."""
    keyword_lower = keyword.lower()
    for col in df.columns:
        if keyword_lower in str(col).lower():
            return col
    return None


def load_and_prepare(csv_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Carrega CSV e prepara dados para análise de Q20 por área de atuação.
    Retorna: (df_clean, info_dict)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None
    
    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Encontrar colunas
    col_area = find_col(df, "Qual é a sua principal área de atuação na engenharia de software no momento?")
    col_q20 = find_col(df, "soft skills")
    
    if col_area is None or col_q20 is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Área de Atuação: {col_area}")
        print(f"   Q20 (Soft Skills): {col_q20}")
        return None, None
    
    print(f"✓ Coluna área de atuação: {repr(col_area)}")
    print(f"✓ Coluna Q20: {repr(col_q20)}")
    
    # Preparar dados
    df_clean = df[[col_area, col_q20]].copy()
    df_clean.columns = ["area", "q20"]
    
    print(f"\n🔍 ANÁLISE DE DADOS REMOVIDOS:")
    print(f"{'=' * 80}")
    total_inicial = len(df_clean)
    print(f"Total inicial: {total_inicial}\n")
    
    # Verificar Q20 inválidos ANTES de mapear
    q20_originais = df_clean["q20"].copy()
    q20_unicos = q20_originais.unique()
    print(f"Valores únicos em Q20 (antes de mapear):")
    for val in sorted(q20_unicos, key=lambda x: str(x)):
        count = (q20_originais == val).sum()
        esta_no_map = val in LIKERT_MAP
        status = "✓ VÁLIDO" if esta_no_map else "✗ INVÁLIDO"
        print(f"   {status}: '{val}' (n={count})")
    
    # Mapear Q20 para numérico
    df_clean["q20"] = df_clean["q20"].map(LIKERT_MAP)
    
    q20_invalidos_mask = df_clean["q20"].isna()
    q20_invalidos_count = q20_invalidos_mask.sum()
    
    if q20_invalidos_count > 0:
        print(f"\n❌ Q20 não mapeados (serão removidos): {q20_invalidos_count}")
        print("   Valores originais:")
        valores_invalidos = q20_originais[q20_invalidos_mask].value_counts()
        for val, count in valores_invalidos.items():
            print(f"      '{val}': {count} ocorrências")
    
    # Verificar áreas inválidas
    area_originais = df_clean["area"].copy()
    area_unicos = area_originais.unique()
    
    print(f"\n📊 Valores únicos em Área de Atuação:")
    for val in sorted(area_unicos, key=lambda x: str(x)):
        count = (area_originais == val).sum()
        print(f"   '{val}' (n={count})")
    
    # Atualizar AREA_ORDER com valores reais do dataset
    global AREA_ORDER
    areas_reais = [a for a in area_originais.dropna().unique() if pd.notna(a)]
    AREA_ORDER = sorted(areas_reais)
    
    print(f"\n✓ AREA_ORDER atualizada com {len(AREA_ORDER)} áreas")
    
    # Filtrar apenas valores conhecidos de área de atuação
    area_antes = len(df_clean)
    df_clean = df_clean[df_clean["area"].isin(AREA_ORDER)].copy()
    area_removidos = area_antes - len(df_clean)
    
    if area_removidos > 0:
        print(f"\n❌ Áreas inválidas (serão removidas): {area_removidos}")
    
    # Remover NaNs (Q20 inválidos após mapeamento)
    nan_antes = len(df_clean)
    df_clean = df_clean.dropna()
    nan_removidos = nan_antes - len(df_clean)
    
    total_final = len(df_clean)
    total_removido = total_inicial - total_final
    
    print(f"\n{'=' * 80}")
    print(f"📊 RESUMO:")
    print(f"   Total inicial: {total_inicial}")
    print(f"   Total final: {total_final}")
    print(f"   Total removido: {total_removido}")
    print(f"   % retido: {(total_final/total_inicial)*100:.1f}%")
    print(f"{'=' * 80}\n")
    
    if len(df_clean) < 10:
        print(f"❌ Dados insuficientes para análise (n={len(df_clean)})")
        return None, None
    
    # Ordenar área como categoria
    df_clean["area"] = pd.Categorical(
        df_clean["area"],
        categories=AREA_ORDER,
        ordered=True
    )
    
    print(f"\n📊 Distribuição por área de atuação:")
    for area in AREA_ORDER:
        count = (df_clean["area"] == area).sum()
        if count > 0:
            mean = df_clean[df_clean["area"] == area]["q20"].mean()
            print(f"   {area}: {count} respondentes, média Q20 = {mean:.2f}")
    
    return df_clean, {
        "col_area": col_area,
        "col_q20": col_q20,
        "n_rows": len(df_clean),
    }


def plot_boxplot_by_area(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot de Q20 por área de atuação com paleta laranja.
    Inclui quadro com estatísticas e significância.
    Áreas ordenadas da menos focada em soft skills para a mais focada.
    Remove áreas com menos de 5 respondentes.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtrar apenas áreas com 5 ou mais respondentes
    contagens = df_clean["area"].value_counts()
    areas_validas = contagens[contagens >= 5].index.tolist()
    df_filtrado = df_clean[df_clean["area"].isin(areas_validas)].copy()
    
    if len(areas_validas) < 2:
        print(f"❌ Apenas {len(areas_validas)} área(s) com n≥5. Análise impossível.")
        return None
    
    print(f"\n✓ Áreas com n≥5: {len(areas_validas)}")
    areas_removidas = len(df_clean["area"].unique()) - len(areas_validas)
    if areas_removidas > 0:
        print(f"❌ Áreas removidas (n<5): {areas_removidas}")
    
    # Atualizar categoria para incluir apenas áreas válidas
    df_filtrado["area"] = pd.Categorical(
        df_filtrado["area"],
        categories=areas_validas,
        ordered=False
    )
    
    # Calcular médias para áreas válidas
    means = df_filtrado.groupby("area", observed=True)["q20"].mean()
    
    # Ordenar áreas por média crescente (menos focado em soft skills -> mais focado)
    areas_ordenadas = sorted(areas_validas, key=lambda x: means[x])
    
    # Gerar paleta laranja para as áreas ordenadas
    n_categories = len(areas_ordenadas)
    orange_palette = _orange_shades(BASE_ORANGE, n_categories)
    
    # Criar dicionário de cores (menos focado = mais escuro, mais focado = mais claro)
    color_dict = {area: orange_palette[i] for i, area in enumerate(areas_ordenadas)}
    
    # Calcular n total
    n_total = len(df_filtrado)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Boxplot com áreas ordenadas por média crescente
    bp = sns.boxplot(
        data=df_filtrado,
        x="area",
        y="q20",
        order=areas_ordenadas,
        hue="area",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    positions = range(len(areas_ordenadas))
    means_ordered = [means[area] for area in areas_ordenadas]
    
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
    for pos, area in enumerate(areas_ordenadas):
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
    
    # Criar rótulos com n de cada grupo
    group_labels = []
    for area in areas_ordenadas:
        n_respondentes = (df_filtrado["area"] == area).sum()
        # Quebrar texto longo
        area_label = area.replace("/", "/\n") if len(area) > 20 else area
        label = f"{area_label}\n(n={n_respondentes})"
        group_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel(f"Área de Atuação (n={n_total})", fontsize=20, fontweight="bold")
    ax.set_xticklabels(group_labels, fontsize=14, rotation=45, ha="right")
    ax.set_ylabel("Q20 (Foco em Soft Skills)", fontsize=20, fontweight="bold")
    # ax.set_title("Distribuição de Q20 por Área de Atuação", fontsize=22, fontweight="bold", pad=20)
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis="both", labelsize=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc="upper left", fontsize=20)
    
    # Calcular estatísticas para o quadro
    grupos = {}
    for area in df_filtrado["area"].unique():
        subset = df_filtrado[df_filtrado["area"] == area]["q20"].values
        if len(subset) >= 5:
            grupos[area] = subset
    
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
            0.98, 0.05,
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


def analise_soft_skills_por_area(csv_path: Optional[Path] = None,
                                 out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, cria boxplot por área de atuação com teste estatístico.
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
    boxplot_path = plot_boxplot_by_area(df_clean, out_dir)
    
    # Estatísticas descritivas
    print(f"\n📊 Estatísticas por Área de Atuação:")
    for area in AREA_ORDER:
        subset = df_clean[df_clean["area"] == area]["q20"]
        if len(subset) > 0:
            print(f"\n  {area}:")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
            print(f"    min = {subset.min():.0f}, max = {subset.max():.0f}")
    
    # Teste de Kruskal-Wallis
    print(f"\n\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q20 são iguais entre todos os grupos de área")
    print(f"Hipótese Alternativa (H1): Pelo menos um grupo tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos por área
    grupos = {}
    for area in df_clean["area"].unique():
        subset = df_clean[df_clean["area"] == area]["q20"].values
        if len(subset) >= 5:  # Filtro: mínimo 5 observações
            grupos[area] = subset
    
    if len(grupos) < 2:
        print(f"❌ Dados insuficientes para teste (menos de 2 grupos com n≥5)")
    else:
        # Executar teste de Kruskal-Wallis
        h_stat, p_valor = stats.kruskal(*grupos.values())
        
        print(f"Número de grupos: {len(grupos)}")
        print(f"Total de observações: {sum(len(v) for v in grupos.values())}\n")
        
        print(f"Estatística H (Kruskal-Wallis): {h_stat:.4f}")
        print(f"P-valor: {p_valor:.6f}")
        print(f"Nível de significância (α): 0.05\n")
        
        # Interpretação
        if p_valor < 0.001:
            sig_level = "***  (Altamente significativo)"
        elif p_valor < 0.01:
            sig_level = "**   (Muito significativo)"
        elif p_valor < 0.05:
            sig_level = "*    (Significativo)"
        else:
            sig_level = "ns   (Não significativo)"
        
        print(f"Resultado: {sig_level}")
        
        if p_valor < 0.05:
            print(f"\n✅ CONCLUSÃO: Rejeita-se H0")
            print(f"   → Existe diferença significativa no foco em soft skills (Q20) entre as áreas de atuação")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa no foco em soft skills (Q20) entre as áreas de atuação")
    
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_groups": df_clean["area"].nunique(),
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_soft_skills_por_area(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Grupos de área: {result['n_groups']}")
        print(f"   Boxplot: {result['boxplot_path']}")