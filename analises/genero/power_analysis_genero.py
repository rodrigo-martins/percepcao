"""
Post-hoc Power Analysis para a comparação de gênero (Masculino vs Feminino).

Responde à pergunta do revisor: "Dado n=216 vs n=65, qual era o poder
estatístico para detectar efeitos de diferentes tamanhos?"

Saídas:
  - Console: resumo completo
  - CSV:     output/power_analysis_gender.csv
"""

from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

LIKERT_MAP = {
    "Concordo totalmente": 5,
    "Concordo": 4,
    "Neutro": 3,
    "Discordo": 2,
    "Discordo totalmente": 1,
}

QUESTION_SHORT = {
    "Q1": "Conexão com objetivos da empresa",
    "Q2": "Apoio e recursos da empresa",
    "Q3": "Conteúdo atendeu necessidades da função",
    "Q4": "Conteúdo adaptado ao nível de experiência",
    "Q5": "Opinião considerada na criação",
    "Q6": "Carga horária prejudicou tempo pessoal",
    "Q7": "Manter-se competitivo no mercado",
    "Q8": "Motivação para buscar conhecimentos",
    "Q9": "Liderança incentivou participação",
    "Q10": "Tempo suficiente no expediente",
    "Q11": "Participou mais por obrigação",
    "Q12": "Incentivos claros da empresa",
    "Q13": "Reconhecimento por concluir treinamentos",
    "Q14": "Bem organizado e estruturado",
    "Q15": "Materiais úteis",
    "Q16": "Ambiente adequado para aprendizado",
    "Q17": "Explicação do instrutor clara",
    "Q18": "Atuação do instrutor fundamental",
    "Q19": "Atividades interessantes e variadas",
    "Q20": "Parte dedicada a soft skills",
    "Q21": "Satisfação com a qualidade",
    "Q22": "Raciocínio para resolver problemas",
    "Q23": "Melhorou desempenho",
    "Q24": "Consigo aplicar no trabalho",
    "Q25": "Mais autonomia no trabalho",
    "Q26": "Suporte da liderança para aplicar",
    "Q27": "Oportunidades de crescimento na empresa",
}


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str, engine="python")

    genero_col = None
    for col in df.columns:
        if "gênero" in str(col).lower() or "genero" in str(col).lower():
            genero_col = col
            break

    if genero_col is None:
        raise ValueError("Coluna de gênero não encontrada")

    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]

    df_clean = df[[genero_col] + likert_cols].copy()
    df_clean = df_clean.rename(columns={genero_col: "genero"})
    df_clean = df_clean[df_clean["genero"].isin(["Masculino", "Feminino"])].copy()

    for col in likert_cols:
        df_clean[col] = df_clean[col].map(LIKERT_MAP)
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df_clean, likert_cols


def run_power_analysis(csv_path: Optional[Path] = None,
                       out_dir: Optional[Path] = None):
    project_root = Path(__file__).resolve().parents[2]
    csv_path = Path(csv_path) if csv_path else project_root / "data" / "tratado.csv"
    out_dir = Path(out_dir) if out_dir else project_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, likert_cols = load_and_prepare(csv_path)

    n_masc_total = (df["genero"] == "Masculino").sum()
    n_fem_total = (df["genero"] == "Feminino").sum()

    # ── 1. Cenários fixos de tamanho de efeito ──────────────────────────
    power_analysis = TTestIndPower()
    ratio = n_fem_total / n_masc_total

    print("=" * 65)
    print("POST-HOC POWER ANALYSIS: GÊNERO")
    print("=" * 65)
    print(f"Homens: n={n_masc_total} | Mulheres: n={n_fem_total} | α=0.05")
    print()

    print("Cenários de tamanho de efeito:")
    for d, label in [(0.2, "pequeno"), (0.5, "médio"), (0.8, "grande")]:
        power = power_analysis.solve_power(
            effect_size=d,
            nobs1=n_masc_total,
            ratio=ratio,
            alpha=0.05,
            alternative="two-sided",
        )
        print(f"  d={d} ({label:7s}): Power = {power * 100:.1f}%")

    # ── 2. Menor efeito detectável com power=0.80 ──────────────────────
    min_d = power_analysis.solve_power(
        nobs1=n_masc_total,
        ratio=ratio,
        alpha=0.05,
        power=0.80,
        alternative="two-sided",
    )
    print(f"\nMenor efeito detectável (power=0.80): d = {min_d:.4f}")

    # ── 3. Interpretação ────────────────────────────────────────────────
    print()
    print("Interpretação:")
    print("  - Para efeitos médios (d≥0.5), a amostra tinha poder adequado.")
    print("    A ausência de significância fortalece o argumento de")
    print("    homogeneidade entre gêneros para esses tamanhos de efeito.")
    print("  - Para efeitos pequenos (d=0.2), o poder é limitado.")
    print("    A ausência de significância não é conclusiva para")
    print("    diferenças sutis entre gêneros.")

    # ── 4. Efeitos observados por item ──────────────────────────────────
    print()
    print("=" * 65)
    print("EFEITOS OBSERVADOS POR ITEM")
    print("=" * 65)

    rows = []
    for i, col in enumerate(likert_cols):
        q_label = f"Q{i + 1}"
        subset = df[["genero", col]].dropna()
        homens = subset[subset["genero"] == "Masculino"][col]
        mulheres = subset[subset["genero"] == "Feminino"][col]

        n_m, n_f = len(homens), len(mulheres)
        mean_m, mean_f = homens.mean(), mulheres.mean()

        # Cohen's d (pooled SD)
        pooled_std = np.sqrt(
            ((n_m - 1) * homens.std(ddof=1) ** 2
             + (n_f - 1) * mulheres.std(ddof=1) ** 2)
            / (n_m + n_f - 2)
        )

        if pooled_std > 0:
            d_obs = abs(mean_m - mean_f) / pooled_std
        else:
            d_obs = 0.0

        # Power para o efeito observado
        if d_obs > 0:
            pwr = power_analysis.solve_power(
                effect_size=d_obs,
                nobs1=n_m,
                ratio=n_f / n_m,
                alpha=0.05,
                alternative="two-sided",
            )
        else:
            pwr = 0.05  # poder = α quando d=0

        # p-valor (Mann-Whitney)
        if n_m >= 5 and n_f >= 5:
            _, p_val = stats.mannwhitneyu(
                homens, mulheres, alternative="two-sided"
            )
        else:
            p_val = np.nan

        short = QUESTION_SHORT.get(q_label, "")
        print(f"{q_label:4s} ({short:42s}): d={d_obs:.4f}, power={pwr * 100:5.1f}%")

        rows.append({
            "item": q_label,
            "label": short,
            "n_masc": n_m,
            "n_fem": n_f,
            "mean_masc": round(mean_m, 4),
            "mean_fem": round(mean_f, 4),
            "sd_pooled": round(pooled_std, 4),
            "cohen_d": round(d_obs, 4),
            "power": round(pwr, 4),
            "p_value_mannwhitney": round(p_val, 6) if not np.isnan(p_val) else np.nan,
        })

    # ── Estatísticas resumidas ──────────────────────────────────────────
    d_values = [r["cohen_d"] for r in rows]
    print()
    print(f"Média dos efeitos observados:  d = {np.mean(d_values):.4f}")
    print(f"Mediana dos efeitos observados: d = {np.median(d_values):.4f}")
    print(f"Máximo efeito observado:        d = {np.max(d_values):.4f}")

    # ── 5. Salvar CSV ───────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    csv_out = out_dir / "power_analysis_gender.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"\nCSV salvo em: {csv_out}")

    return results_df


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_power_analysis(csv_path=csv_arg, out_dir=out_arg)
