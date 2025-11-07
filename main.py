import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Optional

from questions.obrig_optional import analyze_obrig_optional
from questions.idade import analyze_idade 
from questions import analyze_genero  
from questions.mapa_estados import plot_respondentes_por_estado  
from questions.instrucao import analyze_instrucao  
import questions as qs


load_dotenv()

def load_dataframe(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path or os.getenv("DATA_PATH", "data/raw.csv")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV não encontrado: {p}")
    return pd.read_csv(p, sep=",", quotechar='"', engine="python", dtype=str)

def main() -> None:
    df = load_dataframe()
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total de linhas: {len(df)}")

    # descobre e executa todas as funções analyze_* do pacote questions
    for name in sorted(dir(qs)):
        if not name.startswith("analyze_"):
            continue
        func = getattr(qs, name)
        if not callable(func):
            continue
        print(f"\nExecutando {name}() ...")
        try:
            # tenta chamar com out_dir nomeado (padrão nos analisadores)
            res = func(df, out_dir=out_dir)
        except TypeError:
            # fallback: apenas com df
            try:
                res = func(df)
            except Exception as e:
                print(f"Falha ao executar {name}: {e}")
                continue
        except Exception as e:
            print(f"Erro em {name}: {e}")
            continue

        # imprimir resumo padrão se retorno for dicionário compatível
        if isinstance(res, dict):
            col = res.get("column")
            total = res.get("total", 0)
            print(f"-> {name}: coluna={col}, total={total}")
            percentages = res.get("percentages", {})
            for label, (cnt, pct) in percentages.items():
                print(f"   {label}: {cnt} ({pct:.1f}%)")
            img = res.get("image")
            if img:
                print(f"   gráfico: {img}")
        else:
            print(f"{name} retornou: {type(res)}")

if __name__ == "__main__":
    main()