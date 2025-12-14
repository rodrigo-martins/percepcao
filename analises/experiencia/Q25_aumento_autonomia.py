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

# Ordem fixa de experiência
EXPERIENCIA_ORDER = [
    "Até um ano.",
    "Entre 1 e 2 anos.",
    "Entre 3 e 4 anos.",
    "Entre 5 e 6 anos.",
    "Entre 7 e 8 anos.",
    "Mais de 8 anos.",
]


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
    Carrega CSV e prepara dados para análise de Q25 por experiência.
    Retorna: (df_clean, info_dict)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None
    
    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Encontrar colunas
    col_exp = find_col(df, "tempo total de experiência")
    col_q25 = find_col(df, "autonomia")
    
    if col_exp is None or col_q25 is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Experiência: {col_exp}")
        print(f"   Q25 (Autonomia): {col_q25}")
        return None, None
    
    print(f"✓ Coluna experiência: {repr(col_exp)}")
    print(f"✓ Coluna Q25: {repr(col_q25)}")
    
    # Preparar dados
    df_clean = df[[col_exp, col_q25]].copy()
    df_clean.columns = ["experiencia", "q25"]
    
    print(f"\n🔍 ANÁLISE DE DADOS REMOVIDOS:")
    print(f"{'=' * 80}")
    total_inicial = len(df_clean)
    print(f"Total inicial: {total_inicial}\n")
    
    # Verificar Q25 inválidos ANTES de mapear
    q25_originais = df_clean["q25"].copy()
    q25_unicos = q25_originais.unique()
    print(f"Valores únicos em Q25 (antes de mapear):")
    for val in sorted(q25_unicos, key=lambda x: str(x)):
        count = (q25_originais == val).sum()
        esta_no_map = val in LIKERT_MAP
        status = "✓ VÁLIDO" if esta_no_map else "✗ INVÁLIDO"
        print(f"   {status}: '{val}' (n={count})")
    
    # Mapear Q25 para numérico
    df_clean["q25"] = df_clean["q25"].map(LIKERT_MAP)
    
    q25_invalidos_mask = df_clean["q25"].isna()
    q25_invalidos_count = q25_invalidos_mask.sum()
    
    if q25_invalidos_count > 0:
        print(f"\n❌ Q25 não mapeados (serão removidos): {q25_invalidos_count}")
        print("   Valores originais:")
        valores_invalidos = q25_originais[q25_invalidos_mask].value_counts()
        for val, count in valores_invalidos.items():
            print(f"      '{val}': {count} ocorrências")
    
    # Verificar experiências inválidas
    exp_originais = df_clean["experiencia"].copy()
    exp_unicos = exp_originais.unique()
    
    print(f"\n📊 Valores únicos em Experiência:")
    for val in sorted(exp_unicos, key=lambda x: str(x)):
        count = (exp_originais == val).sum()
        esta_na_ordem = val in EXPERIENCIA_ORDER
        status = "✓ VÁLIDO" if esta_na_ordem else "✗ INVÁLIDO"
        print(f"   {status}: '{val}' (n={count})")
    
    # Filtrar apenas valores conhecidos de experiência
    exp_antes = len(df_clean)
    df_clean = df_clean[df_clean["experiencia"].isin(EXPERIENCIA_ORDER)].copy()
    exp_removidos = exp_antes - len(df_clean)
    
    if exp_removidos > 0:
        exp_invalidas_mask = ~exp_originais.isin(EXPERIENCIA_ORDER)
        print(f"\n❌ Experiências inválidas (serão removidas): {exp_removidos}")
        print("   Valores:")
        valores_exp_invalidos = exp_originais[exp_invalidas_mask].value_counts()
        for val, count in valores_exp_invalidos.items():
            print(f"      '{val}': {count} ocorrências")
    
    # Remover NaNs (Q25 inválidos após mapeamento)
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
    
    # Ordenar experiência como categoria
    df_clean["experiencia"] = pd.Categorical(
        df_clean["experiencia"],
        categories=EXPERIENCIA_ORDER,
        ordered=True
    )
    
    print(f"\n📊 Distribuição por experiência:")
    for exp in EXPERIENCIA_ORDER:
        count = (df_clean["experiencia"] == exp).sum()
        if count > 0:
            mean = df_clean[df_clean["experiencia"] == exp]["q25"].mean()
            print(f"   {exp}: {count} respondentes, média Q25 = {mean:.2f}")
    
    return df_clean, {
        "col_exp": col_exp,
        "col_q25": col_q25,
        "n_rows": len(df_clean),
    }


def plot_boxplot_by_experience(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot de Q25 por grupos de experiência com paleta laranja.
    Inclui quadro com estatísticas e significância.
    Experiências ordenadas crescentemente (menor → maior).
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerar paleta laranja (6 tons para 6 categorias)
    n_categories = len(EXPERIENCIA_ORDER)
    orange_palette = _orange_shades(BASE_ORANGE, n_categories)
    
    # Criar dicionário de cores
    color_dict = {exp: orange_palette[i] for i, exp in enumerate(EXPERIENCIA_ORDER)}
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Boxplot
    bp = sns.boxplot(
        data=df_clean,
        x="experiencia",
        y="q25",
        order=EXPERIENCIA_ORDER,
        hue="experiencia",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    means = df_clean.groupby("experiencia", observed=True)["q25"].mean()
    positions = range(len(EXPERIENCIA_ORDER))
    means_ordered = [means[exp] for exp in EXPERIENCIA_ORDER]
    
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
    for pos, exp in enumerate(EXPERIENCIA_ORDER):
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
    abreviacoes = {
        "Até um ano.": "<1",
        "Entre 1 e 2 anos.": "1-2",
        "Entre 3 e 4 anos.": "3-4",
        "Entre 5 e 6 anos.": "5-6",
        "Entre 7 e 8 anos.": "7-8",
        "Mais de 8 anos.": ">8",
    }
    for exp in EXPERIENCIA_ORDER:
        n_respondentes = (df_clean["experiencia"] == exp).sum()
        abrev = abreviacoes.get(exp, exp)
        label = f"{abrev}\n(n={n_respondentes})"
        group_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel("Tempo Total de Experiência", fontsize=20, fontweight="bold")
    ax.set_xticklabels(group_labels, fontsize=16)
    ax.set_ylabel("Q25 (Aumento de autonomia)", fontsize=20, fontweight="bold")
    # ax.set_title(
    #     "Distribuição de Q25 (Autonomia) por Tempo de Experiência",
    #     fontsize=20,
    #     fontweight="bold",
    #     pad=20
    # )
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis="both", labelsize=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)
    
    # Legenda
    ax.legend(loc="upper right", fontsize=20)
    
    # Calcular estatísticas para o quadro
    grupos = {}
    for exp in df_clean["experiencia"].unique():
        subset = df_clean[df_clean["experiencia"] == exp]["q25"].values
        if len(subset) >= 5:
            grupos[exp] = subset
    
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


def analise_autonomia_por_experiencia(csv_path: Optional[Path] = None,
                                      out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, cria boxplot por experiência com teste estatístico.
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
    boxplot_path = plot_boxplot_by_experience(df_clean, out_dir)
    
    # Estatísticas descritivas
    print(f"\n📊 Estatísticas por Experiência:")
    for exp in EXPERIENCIA_ORDER:
        subset = df_clean[df_clean["experiencia"] == exp]["q25"]
        if len(subset) > 0:
            print(f"\n  {exp}:")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
            print(f"    min = {subset.min():.0f}, max = {subset.max():.0f}")
    
    # Teste de Kruskal-Wallis
    print(f"\n\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q25 são iguais entre todos os grupos de experiência")
    print(f"Hipótese Alternativa (H1): Pelo menos um grupo tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos por experiência
    grupos = {}
    for exp in df_clean["experiencia"].unique():
        subset = df_clean[df_clean["experiencia"] == exp]["q25"].values
        if len(subset) >= 5:  # Filtro: mínimo 5 observações
            grupos[exp] = subset
    
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
            print(f"   → Existe diferença significativa na concordância (Q25) entre os grupos de experiência")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa na concordância (Q25) entre os grupos de experiência")
    
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_groups": df_clean["experiencia"].nunique(),
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_autonomia_por_experiencia(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Grupos de experiência: {result['n_groups']}")
        print(f"   Boxplot: {result['boxplot_path']}")