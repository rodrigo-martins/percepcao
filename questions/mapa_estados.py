from pathlib import Path
from typing import Optional
import re

import pandas as pd
import geopandas as gpd


def plot_respondentes_por_estado(
    df: pd.DataFrame,
    out_dir: Optional[Path] = Path("output"),
    state_col: str = "Em que Estado você reside? "
) -> Optional[Path]:
    """
    Gera mapa dos respondentes por estado, SEM escala/barra de cores,
    e adiciona rótulos com valor absoluto + porcentagem por estado.
    Retorna Path do arquivo salvo ou None em caso de erro.
    """
    out_dir = Path(out_dir) if out_dir is not None else Path("output")

    if state_col not in df.columns:
        print(f"Coluna '{state_col}' não encontrada no DataFrame.")
        return None

    # contagem por estado e extração da sigla (UF)
    df_counts = df[state_col].astype(str).value_counts().reset_index()
    df_counts.columns = ["estado", "count"]
    df_counts["UF"] = df_counts["estado"].str.extract(r"\((.*?)\)")[0].str.upper()

    # carregar shapefile dos estados via geobr
    try:
        import geobr
        estados = geobr.read_state()
    except Exception:
        print("Biblioteca 'geobr' não disponível. Instale com: pip install geobr")
        return None

    # normalizar coluna de sigla e juntar
    if "abbrev_state" in estados.columns:
        estados = estados.rename(columns={"abbrev_state": "UF"})
    estados["UF"] = estados["UF"].astype(str).str.upper()

    mapa = estados.merge(df_counts[["UF", "count"]], on="UF", how="left")
    mapa["count"] = mapa["count"].fillna(0).astype(int)

    total = int(mapa["count"].sum())
    # acrescenta coluna de porcentagem para rótulos
    mapa["pct"] = (mapa["count"] / total * 100).fillna(0)

    # importar matplotlib e preparar colormap / normalização
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception:
        print("matplotlib não disponível.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    # plotar sem legenda (sem escala)
    cmap = plt.get_cmap("Oranges")
    norm = mpl.colors.Normalize(vmin=mapa["count"].min(), vmax=mapa["count"].max() if mapa["count"].max() > 0 else 1)
    mapa.plot(
        column="count",
        cmap=cmap,
        linewidth=0.8,
        edgecolor="gray",
        legend=False,  # remover escala/barra de cores
        ax=ax
    )

    ax.set_title("Distribuição de Respondentes por Estado", fontsize=16, fontweight="bold")
    ax.axis("off")

    # anotações: colocar número inteiro em cima e porcentagem abaixo
    for _, row in mapa.iterrows():
        cnt = int(row["count"])
        if cnt <= 0:
            continue
        try:
            pt = row.geometry.representative_point()
            x, y = pt.x, pt.y
            pct = row["pct"]
            # rótulo em duas linhas: número inteiro em cima, porcentagem abaixo
            label = f"{cnt}\n{pct:.1f}%"
            # escolher cor do texto conforme a cor do estado no mapa
            face = cmap(norm(cnt))
            luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            text_color = "white" if luminance < 0.6 else "black"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
        except Exception:
            continue

    out_path = out_dir / "mapa_estados.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# execução direta: python questions/mapa_estados.py [csv_path] [state_column]
if __name__ == "__main__":
    import sys
    import os
    import traceback

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    state_col_arg = sys.argv[2] if len(sys.argv) > 2 else "Em que Estado você reside? "

    p = Path(csv_path)
    if not p.exists():
        print(f"CSV não encontrado: {p}")
        sys.exit(1)

    try:
        df = pd.read_csv(p, dtype=str, engine="python")
    except Exception as e:
        print("Falha ao ler CSV:", e)
        traceback.print_exc()
        sys.exit(1)

    try:
        out_dir = project_root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = plot_respondentes_por_estado(df, out_dir=out_dir, state_col=state_col_arg)
        if out_path:
            print("Mapa salvo em:", out_path)
        else:
            print("Mapa não gerado. Verifique mensagens acima.")
    except Exception as e:
        print("Erro ao gerar o mapa:", e)
        traceback.print_exc()