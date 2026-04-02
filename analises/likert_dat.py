"""
Extrai colunas Likert (entre colchetes) do CSV tratado, converte para
escala numérica 1-5 e salva como .dat (espaço-separado, sem cabeçalho).

Uso: .venv/bin/python analises/likert_dat.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "tratado.csv"
OUTPUT_PATH = PROJECT_ROOT / "output" / "likert.dat"

ESCALA = {
    "Discordo totalmente": 1,
    "Discordo": 2,
    "Neutro": 3,
    "Concordo": 4,
    "Concordo totalmente": 5,
}

def main():
    df = pd.read_csv(CSV_PATH)

    # Selecionar colunas Likert (nome contém colchetes)
    likert_cols = [c for c in df.columns if "[" in c and "]" in c]

    df_likert = df[likert_cols].copy()

    # Converter texto para numérico
    pd.set_option("future.no_silent_downcasting", True)
    df_likert = df_likert.replace(ESCALA)

    # Garantir que tudo é numérico
    df_likert = df_likert.apply(pd.to_numeric, errors="coerce")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_likert.to_csv(OUTPUT_PATH, sep=" ", index=False, header=False)

    print(f"Colunas Likert: {len(likert_cols)}")
    print(f"Respondentes: {len(df_likert)}")
    print(f"Arquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
