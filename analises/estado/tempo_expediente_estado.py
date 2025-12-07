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
    import seaborn as sns
except Exception:
    plt = None
    sns = None
    mcolors = None

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

# Mapeamento de Estado para Sigla
ESTADO_SIGLA = {
    'Acre': 'AC', 'Amapá': 'AP', 'Amazonas': 'AM', 'Pará': 'PA',
    'Rondônia': 'RO', 'Roraima': 'RR', 'Tocantins': 'TO',
    'Alagoas': 'AL', 'Bahia': 'BA', 'Ceará': 'CE',
    'Maranhão': 'MA', 'Paraíba': 'PB', 'Pernambuco': 'PE',
    'Piauí': 'PI', 'Rio Grande do Norte': 'RN', 'Sergipe': 'SE',
    'Distrito Federal': 'DF', 'Goiás': 'GO',
    'Mato Grosso': 'MT', 'Mato Grosso do Sul': 'MS',
    'Espírito Santo': 'ES', 'Minas Gerais': 'MG',
    'Rio de Janeiro': 'RJ', 'São Paulo': 'SP',
    'Paraná': 'PR', 'Rio Grande do Sul': 'RS', 'Santa Catarina': 'SC'
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
    Carrega CSV, mapeia Q10 e Estado para numérico/string.
    Filtra apenas estados com n >= 5 respostas.
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
            starts_bracket = "[" if str(col).strip().startswith("[") else " "
            print(f"   {starts_bracket} {i:2d}: {repr(col)}")
        return None, None
    
    print(f"✓ Q10 coluna: {repr(q10_col)}")
    print(f"✓ Estado coluna: {repr(estado_col)}")
    
    # Preparar dados
    df_clean = df[[estado_col, q10_col]].copy()
    df_clean.columns = ["estado", "q10"]
    
    # Mapear Q10 para numérico
    df_clean["q10"] = df_clean["q10"].map(LIKERT_MAP)
    df_clean["q10"] = pd.to_numeric(df_clean["q10"], errors="coerce")
    
    # Limpar Estado
    df_clean["estado"] = df_clean["estado"].apply(clean_estado)
    
    # Remover NaNs
    df_clean = df_clean.dropna(subset=["estado", "q10"])
    
    print(f"✓ Dados limpos: {len(df_clean)} linhas válidas")
    
    if len(df_clean) < 10:
        print(f"❌ Dados insuficientes para análise (n={len(df_clean)})")
        return None, None
    
    # Converter Estado para Sigla
    df_clean["estado_sigla"] = df_clean["estado"].map(ESTADO_SIGLA)
    
    # Remover estados não mapeados
    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["estado_sigla"])
    n_after = len(df_clean)
    
    print(f"✓ Estados mapeados: {n_after} linhas válidas ({n_before - n_after} não mapeados)")
    
    # Filtrar estados com n >= 5 respostas
    estado_counts = df_clean["estado"].value_counts()
    estados_validos = estado_counts[estado_counts >= 5].index.tolist()
    
    n_before_filter = len(df_clean)
    df_clean = df_clean[df_clean["estado"].isin(estados_validos)]
    n_after_filter = len(df_clean)
    
    print(f"\n✓ Filtro n >= 5: {n_after_filter} linhas válidas ({n_before_filter - n_after_filter} removidas)")
    print(f"✓ Estados mantidos: {len(estados_validos)}")
     
    print(f"\n📊 Distribuição por Estado:")
    for estado in sorted(df_clean["estado"].unique()):
        sigla = ESTADO_SIGLA.get(estado, "??")
        count = (df_clean["estado"] == estado).sum()
        mean = df_clean[df_clean["estado"] == estado]["q10"].mean()
        print(f"   {sigla}: {count} respondentes, média = {mean:.2f}")
    
    return df_clean, {"n_states": df_clean["estado"].nunique()}


def plot_boxplot_states(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot com paleta laranja mostrando distribuição de Q10 por estado.
    Estados ordenados da maior para menor média.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcular médias por estado
    state_means = df_clean.groupby("estado")["q10"].mean().sort_values(ascending=False)
    state_order = state_means.index.tolist()
    
    print(f"\n📈 Ordem dos estados (maior → menor média):")
    for i, estado in enumerate(state_order, 1):
        sigla = ESTADO_SIGLA.get(estado, "??")
        print(f"   {i:2d}. {sigla} ({estado}): {state_means[estado]:.2f}")
    
    # Gerar paleta de laranja (estados em ordem)
    n_states = len(state_order)
    orange_palette = _orange_shades(BASE_ORANGE, n_states)
    
    # Criar dicionário de cores baseado na ordem
    color_dict = {estado: orange_palette[i] for i, estado in enumerate(state_order)}
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plotar boxplot
    bp = sns.boxplot(
        data=df_clean,
        x="estado",
        y="q10",
        order=state_order,
        hue="estado",
        palette=color_dict,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    means = df_clean.groupby("estado", observed=True)["q10"].mean()
    positions = range(len(state_order))
    
    means_ordered = [means[estado] for estado in state_order]
    
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
    # Cor do texto: branco para estados escuros (maior média), preto para claros (Q10)
    for pos, estado in enumerate(state_order):
        mean_val = means_ordered[pos]
        # Primeiros 70% (mais escuros) = branco, resto = preto
        threshold = int(len(state_order) * 0.95)
        text_color = "black" if pos < threshold else "white"
        
        ax.text(
            pos,
            mean_val + 0.15,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=text_color
        )
    
    # Criar rótulos com sigla e n de cada estado
    state_labels = []
    for estado in state_order:
        sigla = ESTADO_SIGLA.get(estado, "??")
        n_respondentes = (df_clean["estado"] == estado).sum()
        label = f"{sigla}\n(n={n_respondentes})"
        state_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel("Estado", fontsize=20, fontweight="bold")
    ax.set_xticklabels(state_labels, fontsize=16)
    ax.set_ylabel("Q10 (Tempo no experiente)", fontsize=20, fontweight="bold")
    # ax.set_title(
    #     "Distribuição por Estado: Q10 (Tempo no experiente)",
    #     fontsize=18,
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
    for estado in df_clean["estado"].unique():
        subset = df_clean[df_clean["estado"] == estado]["q10"].values
        if len(subset) >= 5:
            grupos[estado] = subset
    
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
            f"H = {h_stat:.4f}\n"
            f"p = {p_valor:.6f}\n"
            f"Resultado: {sig_symbol} {sig_text}"
        )
        
        # Adicionar quadro ao gráfico
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(
            0.02, 0.05,
            quadro_text,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='bottom',
            fontfamily='monospace',
            bbox=props
        )
    
    plt.tight_layout()
    
    # Salvar figura
    out_path = out_dir / "state_analysis_Q10.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"\n✅ Boxplot salvo em: {out_path}")
    return out_path


def analise_competitividade_estado(csv_path: Optional[Path] = None,
                                    out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, ordena por estado, plota boxplot e realiza teste de Kruskal-Wallis para Q10.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    csv_path = Path(csv_path) if csv_path is not None else project_root / "data" / "ordered.csv"
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
    boxplot_path = plot_boxplot_states(df_clean, out_dir)
    
    # Estatísticas descritivas (top 10)
    print(f"\n📊 Estatísticas Descritivas (Top 10 Estados - Q10):")
    for estado in sorted(df_clean["estado"].unique())[:10]:
        subset = df_clean[df_clean["estado"] == estado]["q10"]
        sigla = ESTADO_SIGLA.get(estado, "??")
        if len(subset) > 0:
            print(f"\n  {sigla} ({estado}):")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
    
    # Teste de Kruskal-Wallis
    print(f"\n\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q10 são iguais entre todos os estados")
    print(f"Hipótese Alternativa (H1): Pelo menos um estado tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos por estado
    grupos = {}
    for estado in df_clean["estado"].unique():
        subset = df_clean[df_clean["estado"] == estado]["q10"].values
        if len(subset) >= 5:  # Filtro: mínimo 5 observações
            grupos[estado] = subset
    
    if len(grupos) < 2:
        print(f"❌ Dados insuficientes para teste (menos de 2 grupos com n≥5)")
    else:
        # Executar teste de Kruskal-Wallis
        h_stat, p_valor = stats.kruskal(*grupos.values())
        
        print(f"Número de estados (grupos): {len(grupos)}")
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
            print(f"   → Existe diferença significativa na concordância (Q10) entre os estados")
            print(f"   → Os estados diferem em relação ao grau de concordância sobre")
            print(f"     ter tempo suficiente durante o expediente")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa na concordância (Q10) entre os estados")
            print(f"   → Os estados têm distribuições semelhantes de resposta")
        
        # Salvar resultados em arquivo
        results_path = out_dir / "kruskal_wallis_Q10.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("TESTE DE KRUSKAL-WALLIS - Q10 por Estado\n")
            f.write("=" * 80 + "\n\n")
            f.write("Hipótese Nula (H0): As distribuições de Q10 são iguais entre todos os estados\n")
            f.write("Hipótese Alternativa (H1): Pelo menos um estado tem distribuição diferente\n\n")
            f.write(f"Estatística H: {h_stat:.4f}\n")
            f.write(f"P-valor: {p_valor:.6f}\n")
            f.write(f"Nível de significância (α): 0.05\n")
            f.write(f"Número de estados: {len(grupos)}\n")
            f.write(f"Total de observações: {sum(len(v) for v in grupos.values())}\n\n")
            f.write(f"Resultado: {sig_level}\n\n")
            
            if p_valor < 0.05:
                f.write("CONCLUSÃO: Rejeita-se H0\n")
                f.write("→ Existe diferença significativa na concordância (Q10) entre os estados\n")
            else:
                f.write("CONCLUSÃO: Falha em rejeitar H0\n")
                f.write("→ Não há diferença significativa na concordância (Q10) entre os estados\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("ESTATÍSTICAS DESCRITIVAS POR ESTADO (Q10)\n")
            f.write("=" * 80 + "\n\n")
            
            for estado in sorted(grupos.keys()):
                sigla = ESTADO_SIGLA.get(estado, "??")
                subset = grupos[estado]
                f.write(f"{sigla} ({estado}):\n")
                f.write(f"  n = {len(subset)}\n")
                f.write(f"  Média = {subset.mean():.2f}\n")
                f.write(f"  Mediana = {np.median(subset):.2f}\n")
                f.write(f"  Std = {subset.std():.2f}\n")
                f.write(f"  Min = {subset.min():.0f}, Max = {subset.max():.0f}\n\n")
        
        print(f"\n📄 Resultados salvos em: {results_path}")
    
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_states": prep_info["n_states"],
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
    }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_competitividade_estado(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Estados: {result['n_states']}")
        print(f"   Boxplot: {result['boxplot_path']}")