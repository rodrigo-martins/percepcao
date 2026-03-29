# filepath: analises/confiabilidade.py
"""
Alfa de Cronbach e Ômega de McDonald por dimensão (fatores da AFE).
Uso: .venv/bin/python analises/confiabilidade.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from factor_analyzer import FactorAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIKERT_PATH = PROJECT_ROOT / "output" / "likert.dat"
OUTPUT_DIR = PROJECT_ROOT / "output" / "confiabilidade"

# Dimensões derivadas da AFE (varimax, 5 fatores)
# Mapeamento: nome da dimensão -> lista de itens
DIMENSOES = {
    "Impacto e Aplicação": ["Q7", "Q8", "Q22", "Q23", "Q24", "Q25"],
    "Qualidade Instrucional": ["Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q21"],
    "Suporte Organizacional": ["Q9", "Q10", "Q12", "Q13", "Q26", "Q27"],
    "Personalização e Conteúdo": ["Q3", "Q4", "Q5", "Q20"],
    "Contexto e Recursos": ["Q1", "Q2", "Q6"],
}

# Q6 e Q11 são itens reversos (escala invertida)
ITENS_REVERSOS = ["Q6", "Q11"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(LIKERT_PATH, sep=r"\s+", header=None, engine="python")
    df.columns = [f"Q{i+1}" for i in range(df.shape[1])]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    # inverter itens reversos (escala 1-5 -> 6-x)
    for col in ITENS_REVERSOS:
        if col in df.columns:
            df[col] = 6 - df[col]
    return df


def cronbach_alpha(df: pd.DataFrame) -> tuple[float, list[float]]:
    """Retorna (alpha, alpha_if_dropped para cada item)."""
    alpha = pg.cronbach_alpha(df)[0]
    alpha_dropped = []
    for col in df.columns:
        remaining = df.drop(columns=[col])
        a = pg.cronbach_alpha(remaining)[0]
        alpha_dropped.append(round(a, 4))
    return round(alpha, 4), alpha_dropped


def omega_mcdonald(df: pd.DataFrame) -> float:
    """
    Ômega de McDonald (omega total) calculado via modelo unifatorial.
    omega = (sum(loadings))^2 / ((sum(loadings))^2 + sum(uniquenesses))
    """
    if df.shape[1] < 2:
        return np.nan
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(df)
    loadings = fa.loadings_.flatten()
    uniquenesses = fa.get_uniquenesses()
    sum_l = loadings.sum()
    omega = sum_l**2 / (sum_l**2 + uniquenesses.sum())
    return round(float(omega), 4)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    results = []
    details = []

    print("=" * 70)
    print("CONSISTÊNCIA INTERNA POR DIMENSÃO")
    print("=" * 70)

    for dim_name, itens in DIMENSOES.items():
        subset = df[itens]
        alpha, alpha_dropped = cronbach_alpha(subset)
        omega = omega_mcdonald(subset)

        print(f"\n--- {dim_name} ---")
        print(f"  Itens: {', '.join(itens)} (n={len(itens)})")
        print(f"  Cronbach α = {alpha:.4f}")
        print(f"  McDonald ω = {omega:.4f}")
        print(f"  Alpha se item removido:")
        for col, a_d in zip(itens, alpha_dropped):
            flag = " ← melhora" if a_d > alpha else ""
            print(f"    {col}: α = {a_d:.4f}{flag}")

        results.append({
            "Dimensão": dim_name,
            "Itens": ", ".join(itens),
            "n_itens": len(itens),
            "Cronbach_alpha": alpha,
            "McDonald_omega": omega,
        })

        for col, a_d in zip(itens, alpha_dropped):
            details.append({
                "Dimensão": dim_name,
                "Item": col,
                "Alpha_se_removido": a_d,
            })

    # Escala global (todos os 27 itens)
    all_items = [q for qs in DIMENSOES.values() for q in qs]
    # adicionar itens que não estão em nenhuma dimensão (Q11 ficou fora por carga cruzada)
    all_likert = [f"Q{i+1}" for i in range(df.shape[1])]
    missing = [q for q in all_likert if q not in all_items]
    global_items = all_items + missing
    subset_global = df[global_items]
    alpha_g, _ = cronbach_alpha(subset_global)
    omega_g = omega_mcdonald(subset_global)

    print(f"\n{'=' * 70}")
    print(f"ESCALA GLOBAL ({len(global_items)} itens)")
    print(f"  Cronbach α = {alpha_g:.4f}")
    print(f"  McDonald ω = {omega_g:.4f}")
    print("=" * 70)

    results.append({
        "Dimensão": "GLOBAL",
        "Itens": f"{len(global_items)} itens",
        "n_itens": len(global_items),
        "Cronbach_alpha": alpha_g,
        "McDonald_omega": omega_g,
    })

    # Salvar CSVs
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "confiabilidade_resumo.csv", index=False)

    df_details = pd.DataFrame(details)
    df_details.to_csv(OUTPUT_DIR / "confiabilidade_detalhes.csv", index=False)

    print(f"\nArquivos salvos em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
