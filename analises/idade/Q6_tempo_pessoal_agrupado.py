from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

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

# Mapeamento da escala Likert
LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}


def _orange_shades(base_hex: str, n: int) -> list:
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
    Carrega CSV, mapeia Q6 e Idade para numérico, cria grupos: Q1, Q2+Q3, Q4.
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
        
        # Procurar por Q6 (carga horária)
        if col_lower.startswith("[") and "carga horária" in col_lower and "prejudicou" in col_lower:
            q6_col = col
        
        # Procurar por "Qual é a sua idade?"
        if not col_lower.startswith("[") and "qual" in col_lower and "idade" in col_lower:
            age_col = col
    
    if q6_col is None or age_col is None:
        print(f"❌ Colunas não encontradas!")
        print(f"   Q6 (carga horária): {q6_col}")
        print(f"   Idade: {age_col}")
        return None, None
    
    print(f"✓ Q6 coluna: {repr(q6_col)}")
    print(f"✓ Idade coluna: {repr(age_col)}")
    
    # Preparar dados
    df_clean = df[[age_col, q6_col]].copy()
    df_clean.columns = ["idade", "q6"]
    
    # Mapear Q6 para numérico
    df_clean["q6"] = df_clean["q6"].map(LIKERT_MAP)
    df_clean["q6"] = pd.to_numeric(df_clean["q6"], errors="coerce")
    
    # Converter Idade para numérico
    df_clean["idade"] = df_clean["idade"].apply(convert_age_to_numeric)
    
    # Remover NaNs
    df_clean = df_clean.dropna(subset=["idade", "q6"])
    
    print(f"✓ Dados limpos: {len(df_clean)} linhas válidas")
    
    if len(df_clean) < 10:
        print(f"❌ Dados insuficientes para análise (n={len(df_clean)})")
        return None, None
    
    # Criar quartis de idade
    quartiles = pd.qcut(
        df_clean["idade"],
        q=4,
        labels=False,
        duplicates="drop"
    )
    
    # Agrupar: Q1, Q2+Q3, Q4
    def group_quartiles(q):
        if q == 0:
            return "Q1"
        elif q == 1 or q == 2:
            return "Q2+Q3"
        else:
            return "Q4"
    
    df_clean["Faixa_Etaria"] = quartiles.apply(group_quartiles)
    
    # Calcular intervalos reais de idade para cada grupo
    group_ranges = {}
    for group in ["Q1", "Q2+Q3", "Q4"]:
        mask = df_clean["Faixa_Etaria"] == group
        if mask.any():
            min_age = df_clean.loc[mask, "idade"].min()
            max_age = df_clean.loc[mask, "idade"].max()
            group_ranges[group] = f"{int(min_age)}-{int(max_age)}"
    
    # Reordenar grupos na ordem correta
    group_order = ["Q1", "Q2+Q3", "Q4"]
    df_clean["Faixa_Etaria"] = pd.Categorical(
        df_clean["Faixa_Etaria"],
        categories=group_order,
        ordered=True
    )
    
    print(f"\n📊 Grupos de Idade criados:")
    for group in group_order:
        count = (df_clean["Faixa_Etaria"] == group).sum()
        age_range = group_ranges.get(group, "N/A")
        print(f"   {group} ({age_range}): {count} respondentes")
    
    return df_clean, {"n_groups": len(group_order), "ranges": group_ranges}


def plot_boxplot_quartis(df_clean: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """
    Cria boxplot com paleta laranja mostrando distribuição de Q6 por grupos de idade.
    Grupos: Q1 (escuro), Q2+Q3 (médio), Q4 (claro)
    Inclui quadro com teste estatístico de significância.
    """
    if plt is None or sns is None:
        print("❌ matplotlib/seaborn não disponível")
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerar paleta de laranja (3 tons para 3 grupos)
    n_groups = 3
    orange_palette = _orange_shades(BASE_ORANGE, n_groups)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plotar boxplot com Q6
    bp = sns.boxplot(
        data=df_clean,
        x="Faixa_Etaria",
        y="q6",
        hue="Faixa_Etaria",
        palette=orange_palette,
        legend=False,
        ax=ax,
        width=0.6,
        linewidth=2.5,
    )
    
    # Calcular e plotar médias como bolinhas pretas
    means = df_clean.groupby("Faixa_Etaria", observed=True)["q6"].mean()
    positions = range(len(means))
    
    ax.scatter(
        positions,
        means.values,
        color="black",
        s=150,
        zorder=3,
        marker="o",
        edgecolor="white",
        linewidth=2,
        label="Média"
    )
    
    # Adicionar valores das médias como texto
    # Q1 (escuro) = branco, Q2+Q3 (médio) = branco, Q4 (claro) = preto
    text_colors = {"Q1": "black", "Q2+Q3": "black", "Q4": "black"}
    
    for pos, (group, mean_val) in enumerate(means.items()):
        text_color = text_colors.get(group, "black")
        
        ax.text(
            pos,
            mean_val + 0.25,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color=text_color
        )
    
    # Obter intervalos de idade para cada grupo
    group_labels = []
    for group in ["Q1", "Q2+Q3", "Q4"]:
        mask = df_clean["Faixa_Etaria"] == group
        if mask.any():
            min_age = df_clean.loc[mask, "idade"].min()
            max_age = df_clean.loc[mask, "idade"].max()
            n_respondentes = mask.sum()
            label = f"{group} ({int(min_age)}-{int(max_age)})\n(n={n_respondentes})"
            group_labels.append(label)
    
    # Configurar títulos e labels
    ax.set_xlabel("Grupos de Idade", fontsize=20, fontweight="bold")
    ax.set_xticklabels(group_labels, fontsize=14)
    ax.set_ylabel("Q6 (Impacto no tempo pessoal)", fontsize=20, fontweight="bold")
    # ax.set_title(
    #     "Distribuição por Grupos de Idade: Q6 (Carga horária)",
    #     fontsize=18,
    #     fontweight="bold",
    #     pad=20
    # )
    
    # Aumentar tamanho dos ticks
    ax.tick_params(axis="both", labelsize=18)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 5.5)
    
    # Legenda
    ax.legend(loc="upper right", fontsize=16)
    
    # Calcular estatísticas para o quadro
    grupos = {}
    for group in ["Q1", "Q2+Q3", "Q4"]:
        subset = df_clean[df_clean["Faixa_Etaria"] == group]["q6"].values
        if len(subset) >= 5:
            grupos[group] = subset
    
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
            0.02, 0.95,
            quadro_text,
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment='top',
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


def analise_idade_vs_tempo_grupo(csv_path: Optional[Path] = None,
                                  out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo: carrega dados, cria grupos e plota boxplot com teste estatístico.
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
    boxplot_path = plot_boxplot_quartis(df_clean, out_dir)
    
    # Criar quadro com estatísticas por grupo
    print(f"\n📋 QUADRO DE ESTATÍSTICAS DESCRITIVAS POR GRUPO (Q6)")
    print(f"{'=' * 100}")
    
    stats_data = []
    for group in ["Q1", "Q2+Q3", "Q4"]:
        subset = df_clean[df_clean["Faixa_Etaria"] == group]["q6"]
        if len(subset) > 0:
            stats_data.append({
                "Grupo": group,
                "n": len(subset),
                "Média": f"{subset.mean():.2f}",
                "Mediana": f"{subset.median():.2f}",
                "Desvio Padrão": f"{subset.std():.2f}",
                "Mín": f"{subset.min():.0f}",
                "Máx": f"{subset.max():.0f}",
            })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    print(f"{'=' * 100}\n")
    
    # Teste de Kruskal-Wallis
    print(f"\n🔬 TESTE DE KRUSKAL-WALLIS (ANOVA Não-paramétrica)")
    print(f"{'=' * 80}")
    print(f"Hipótese Nula (H0): As distribuições de Q6 são iguais entre todos os grupos")
    print(f"Hipótese Alternativa (H1): Pelo menos um grupo tem distribuição diferente")
    print(f"{'=' * 80}\n")
    
    # Preparar grupos
    grupos = {}
    for group in ["Q1", "Q2+Q3", "Q4"]:
        subset = df_clean[df_clean["Faixa_Etaria"] == group]["q6"]
        if len(subset) > 0:
            grupos[group] = subset.values
            print(f"\n  {group}:")
            print(f"    n = {len(subset)}")
            print(f"    média = {subset.mean():.2f}")
            print(f"    mediana = {subset.median():.2f}")
            print(f"    std = {subset.std():.2f}")
            print(f"    min = {subset.min():.0f}, max = {subset.max():.0f}")
    
    if len(grupos) >= 2:
        # Executar teste de Kruskal-Wallis
        h_stat, p_valor = stats.kruskal(*grupos.values())
        
        print(f"\n\nNúmero de grupos: {len(grupos)}")
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
            print(f"   → Existe diferença significativa na percepção (Q6) entre os grupos de idade")
            print(f"   → Os grupos de idade diferem em relação ao grau de concordância sobre")
            print(f"     carga horária prejudicar tempo pessoal")
        else:
            print(f"\n❌ CONCLUSÃO: Falha em rejeitar H0")
            print(f"   → Não há diferença significativa na percepção (Q6) entre os grupos de idade")
            print(f"   → Os grupos de idade têm distribuições semelhantes de resposta")
        
        # Salvar resultados em arquivo
        results_path = out_dir / "kruskal_wallis_idade_q6.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("TESTE DE KRUSKAL-WALLIS - Q6 por Grupos de Idade\n")
            f.write("=" * 80 + "\n\n")
            f.write("Hipótese Nula (H0): As distribuições de Q6 são iguais entre todos os grupos\n")
            f.write("Hipótese Alternativa (H1): Pelo menos um grupo tem distribuição diferente\n\n")
            
            # Quadro de estatísticas descritivas
            f.write("QUADRO DE ESTATÍSTICAS DESCRITIVAS POR GRUPO\n")
            f.write("-" * 80 + "\n")
            f.write(stats_df.to_string(index=False))
            f.write("\n\n")
            
            # Quadro de teste estatístico
            f.write("QUADRO DE TESTE ESTATÍSTICO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Teste Estatístico: Kruskal-Wallis\n")
            f.write(f"Estatística H: {h_stat:.4f}\n")
            f.write(f"P-valor: {p_valor:.6f}\n")
            f.write(f"Nível de significância (α): 0.05\n")
            f.write(f"Resultado: {sig_symbol}\n")
            f.write(f"Interpretação: {sig_level}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("LEGENDA DE SIGNIFICÂNCIA\n")
            f.write("=" * 80 + "\n")
            f.write("ns  = Não significativo (p ≥ 0.05)\n")
            f.write("*   = Significativo (p < 0.05)\n")
            f.write("**  = Muito significativo (p < 0.01)\n")
            f.write("*** = Altamente significativo (p < 0.001)\n\n")
            
            if p_valor < 0.05:
                f.write("CONCLUSÃO: Rejeita-se H0\n")
                f.write("→ Existe diferença significativa na percepção (Q6) entre os grupos de idade\n")
            else:
                f.write("CONCLUSÃO: Falha em rejeitar H0\n")
                f.write("→ Não há diferença significativa na percepção (Q6) entre os grupos de idade\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("ESTATÍSTICAS DESCRITIVAS DETALHADAS POR GRUPO\n")
            f.write("=" * 80 + "\n\n")
            
            for group in ["Q1", "Q2+Q3", "Q4"]:
                subset = grupos.get(group)
                if subset is not None:
                    f.write(f"{group}:\n")
                    f.write(f"  n = {len(subset)}\n")
                    f.write(f"  Média = {subset.mean():.2f}\n")
                    f.write(f"  Mediana = {np.median(subset):.2f}\n")
                    f.write(f"  Std = {subset.std():.2f}\n")
                    f.write(f"  Min = {subset.min():.0f}, Max = {subset.max():.0f}\n\n")
        
        print(f"\n📄 Resultados salvos em: {results_path}")
     
    return {
        "status": "ok",
        "n_total": len(df_clean),
        "n_groups": prep_info["n_groups"],
        "boxplot_path": boxplot_path,
        "df_clean": df_clean,
     }


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analise_idade_vs_tempo_grupo(csv_path=csv_arg, out_dir=out_arg)
    
    if "error" in result:
        print(f"\n❌ Erro: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Análise concluída com sucesso!")
        print(f"   Total de respondentes: {result['n_total']}")
        print(f"   Quartis: {result['n_groups']}")
        print(f"   Boxplot: {result['boxplot_path']}")