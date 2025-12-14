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

# Ordem fixa de nível profissional (atualizada conforme dados reais)
NIVEL_ORDER = [
    "Em treinamento",
    "Júnior",
    "Pleno",
    "Sênior",
    "Especialista (Foco técnico)",
    "Gerente (Foco em pessoas)",
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
    Carrega CSV e prepara dados para análise de Q27 por nível profissional.
    Retorna: (df_clean, info_dict)
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return None, None
    
    print(f"✓ CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Encontrar colunas
    col_nivel = find_col(df, "nível profissional")
    col_q27 = find_col(df, "oportunidades de crescimento")
    
    if col_nivel is None or col_q27 is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Nível Profissional: {col_nivel}")
        print(f"   Q27 (Crescimento): {col_q27}")
        return None, None
    
    print(f"✓ Coluna nível profissional: {repr(col_nivel)}")
    print(f"✓ Coluna Q27: {repr(col_q27)}")
    
    # Preparar dados
    df_clean = df[[col_nivel, col_q27]].copy()
    df_clean.columns = ["nivel", "q27"]
    
    print(f"\n🔍 ANÁLISE DE DADOS REMOVIDOS:")
    print(f"{'=' * 80}")
    total_inicial = len(df_clean)
    print(f"Total inicial: {total_inicial}\n")
    
    # Verificar Q27 inválidos ANTES de mapear
    q27_originais = df_clean["q27"].copy()
    q27_unicos = q27_originais.unique()
    print(f"Valores únicos em Q27 (antes de mapear):")
    for val in sorted(q27_unicos, key=lambda x: str(x)):
        count = (q27_originais == val).sum()
        esta_no_map = val in LIKERT_MAP
        status = "✓ VÁLIDO" if esta_no_map else "✗ INVÁLIDO"
        print(f"   {status}: '{val}' (n={count})")
    
    # Mapear Q27 para numérico
    df_clean["q27"] = df_clean["q27"].map(LIKERT_MAP)
    
    q27_invalidos_mask = df_clean["q27"].isna()
    q27_invalidos_count = q27_invalidos_mask.sum()
    
    if q27_invalidos_count > 0:
        print(f"\n❌ Q27 não mapeados (serão removidos): {q27_invalidos_count}")
        print("   Valores originais:")
        valores_invalidos = q27_originais[q27_invalidos_mask].value_counts()
        for val, count in valores_invalidos.items():
            print(f"      '{val}': {count} ocorrências")
    
    # Verificar níveis inválidos
    nivel_originais = df_clean["nivel"].copy()
    nivel_unicos = nivel_originais.unique()
    
    print(f"\n📊 Valores únicos em Nível Profissional:")
    for val in sorted(nivel_unicos, key=lambda x: str(x)):
        count = (nivel_originais == val).sum()
        esta_na_ordem = val in NIVEL_ORDER
        status = "✓ VÁLIDO" if esta_na_ordem else "✗ INVÁLIDO"
        print(f"   {status}: '{val}' (n={count})")
    
    # Filtrar apenas valores conhecidos de nível profissional
    nivel_antes = len(df_clean)
    df_clean = df_clean[df_clean["nivel"].isin(NIVEL_ORDER)].copy()
    nivel_removidos = nivel_antes - len(df_clean)
    
    if nivel_removidos > 0:
        nivel_invalidas_mask = ~nivel_originais.isin(NIVEL_ORDER)
        print(f"\n❌ Níveis inválidos (serão removidos): {nivel_removidos}")
        print("   Valores:")
        valores_nivel_invalidos = nivel_originais[nivel_invalidas_mask].value_counts()
        for val, count in valores_nivel_invalidos.items():
            print(f"      '{val}': {count} ocorrências")
    
    # Remover NaNs (Q27 inválidos após mapeamento)
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
    
    # Ordenar nível como categoria
    df_clean["nivel"] = pd.Categorical(
        df_clean["nivel"],
        categories=NIVEL_ORDER,
        ordered=True
    )
    
    print(f"\n📊 Distribuição por nível profissional:")
    for nivel in NIVEL_ORDER:
        count = (df_clean["nivel"] == nivel).sum()
        if count > 0:
            mean = df_clean[df_clean["nivel"] == nivel]["q27"].mean()
            print(f"   {nivel}: {count} respondentes, média Q27 = {mean:.2f}")
    
    return df_clean, {
        "col_nivel": col_nivel,
        "col_q27": col_q27,
        "n_rows": len(df_clean),
    }


def plot_boxplot_by_nivel(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot de Q27 por nível profissional com paleta laranja.
    Inclui quadro com estatísticas e significância.
    Níveis ordenados crescentemente.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtrar apenas níveis com dados
    niveis_com_dados = [n for n in NIVEL_ORDER if (df_clean["nivel"] == n).sum() > 0]
    if len(niveis_com_dados) < 2:
        print(f"❌ Apenas {len(niveis_com_dados)} nível(is) com dados. Análise impossível.")
        return None
    
    # Gerar paleta laranja para TODOS os níveis (não apenas os com dados)
    n_categories = len(NIVEL_ORDER)
    orange_palette = _orange_shades(BASE_ORANGE, n_categories)
    
    # Criar dicionário de cores para TODOS os níveis de NIVEL_ORDER
    color_dict = {nivel: orange_palette[i] for i, nivel in enumerate(NIVEL_ORDER)}
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Boxplot
    bp = sns.boxplot(
        data=df_clean,
        x="nivel",
        y="q27",
        order=niveis_com_dados,
        hue="nivel",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    means = df_clean.groupby("nivel", observed=True)["q27"].mean()
    positions = range(len(niveis_com_dados))
    
    means_ordered = [means[nivel] for nivel in niveis_com_dados]
    
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
    for pos, nivel in enumerate(niveis_com_dados):
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
        "Em treinamento": "Em treinamento",
        "Júnior": "Júnior",
        "Pleno": "Pleno",
        "Sênior": "Sênior",
        "Especialista (Foco técnico)": "Especialista\n(Foco técnico)",
        "Gerente (Foco em pessoas)": "Gerente\n(Foco em pessoas)",
        "Prefiro não responder": "Prefiro não\nresponder",
    }
    for nivel in niveis_com_dados:
        n_respondentes = (df_clean["nivel"] == nivel).sum()
        abrev = abreviacoes.get(nivel, nivel)
        label = f"{abrev}\n(n={n_respondentes})"
        group_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel("Nível Profissional", fontsize=20, fontweight="bold")
    ax.set_xticklabels(group_labels, fontsize=16, rotation=0, ha="center")
    ax.set_ylabel("Q27 (Oportunidades de crescimento)", fontsize=20, fontweight="bold")
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis="both", labelsize=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc="upper right", fontsize=20)
    
    # Calcular estatísticas para o quadro
    grupos = {}
    for nivel in df_clean["nivel"].unique():
        subset = df_clean[df_clean["nivel"] == nivel]["q27"].values
        if len(subset) >= 5:
            grupos[nivel] = subset
    
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
            0.02, 0.98,
            quadro_text,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            horizontalalignment='left',
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


def analise_crescimento_por_nivel(csv_path: Optional[Path] = None,
                                  out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, cria boxplot por nível profissional com teste estatístico.
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
    boxplot_path = plot_boxplot_by_nivel(df_clean, out_dir)
    
    # Estatísticas descritivas
    print(f"\n📊 Estatísticas por Nível Profissional:")
    for nivel in NIVEL_ORDER:
        subset = df_clean[df_clean["nivel"] == nivel]["q27"]
        if len(subset) > 0:
            print(f"\n  {nivel}:")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
            print(f"    min = {subset.min():.0f}, max = {subset.max():.0f}")
    
    # Teste de Kruskal-Wallis
    print(f"\n\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q27 são iguais entre todos os grupos de nível")
    print(f"Hipótese Alternativa (H1): Pelo menos um grupo tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos por nível
    grupos = {}
    for nivel in df_clean["nivel"].unique():
        subset = df_clean[df_clean["nivel"] == nivel]["q27"].values
        if len(subset) >= 5:  # Filtro: mínimo 5 observações
            grupos[nivel] = subset
    
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
            print(f"   → Existe diferença significativa na concordância (Q27) entre os grupos de nível")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa na concordância (Q27) entre os grupos de nível")
    
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_groups": df_clean["nivel"].nunique(),
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_crescimento_por_nivel(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Grupos de nível: {result['n_groups']}")
        print(f"   Boxplot: {result['boxplot_path']}")