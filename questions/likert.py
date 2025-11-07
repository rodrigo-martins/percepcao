from pathlib import Path
from typing import Optional, Dict, Any, List
import re
from textwrap import wrap

import pandas as pd

# plotting libs (graceful fallback)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except Exception:
    plt = None
    mcolors = None
    np = None

BASE_ORANGE = "#ff6002"


def _orange_shades(base_hex: str, n: int) -> List:
    """Gera n tons do laranja base: menor -> mais claro, maior -> mais escuro."""
    try:
        base_rgb = np.array(mcolors.to_rgb(base_hex))
    except Exception:
        return [base_hex] * max(1, n)
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


def analyze_likert(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Detecta colunas Likert (colunas cujo cabeçalho começa com '['), gera gráfico divergent
    (negativo à esquerda, positivo à direita) e salva em output/likert.png.
    Retorna dict compatível com os outros analisadores.
    """
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # localizar colunas Likert (cabecalho iniciando com '[')
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]
    if not likert_cols:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    df_likert = df[likert_cols].copy()

    # limpar cabeçalhos e quebrar texto longo para exibição
    cleaned = [re.sub(r"\[|\]", "", str(c)).strip() for c in df_likert.columns]
    cleaned = ["\n".join(wrap(c, width=70)) for c in cleaned]
    df_likert.columns = cleaned

    # ordem esperada das respostas (mantida)
    order = ["Discordo totalmente", "Discordo", "Neutro", "Concordo", "Concordo totalmente"]

    # contagens por pergunta (linhas = pergunta, colunas = respostas)
    counts = df_likert.apply(lambda x: x.value_counts()).fillna(0).reindex(order).T
    # converter para inteiros
    counts = counts.astype(int)

    # porcentagens por linha
    percent = counts.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=1)

    # inverter sinais dos itens negativos para plot divergente
    for neg in ["Discordo totalmente", "Discordo", "Neutro"]:
        if neg in percent.columns:
            percent[neg] = -percent[neg]

    image_path: Optional[Path] = None
    if plt is None:
        return {
            "column": ", ".join(likert_cols),
            "counts": counts,
            "total": int(len(df)),
            "percentages": {},
            "image": None,
        }

    try:
        # cores: usar tons de laranja do projeto e cinza para neutro
        n_shades = 5
        shades = _orange_shades(BASE_ORANGE, n_shades)
        # mapear categorias para cores (extremos mais escuros)
        # shades index: 0..4 (claros->escuros)
        color_map = {
            "Discordo totalmente": shades[1] if len(shades) > 1 else BASE_ORANGE,
            "Discordo": shades[0] if len(shades) > 0 else BASE_ORANGE,
            "Neutro": (0.827, 0.827, 0.827),  # light gray
            "Concordo": shades[-2] if len(shades) > 1 else BASE_ORANGE,
            "Concordo totalmente": shades[-1] if len(shades) > 0 else BASE_ORANGE,
        }

        # plot stacked horizontal divergent bars
        fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(counts))))
        height = 0.6

        # negativo: Discordo totalmente, Discordo, Neutro (neutro fica à esquerda por escolha)
        neg_cols = [c for c in ["Neutro", "Discordo","Discordo totalmente"] if c in percent.columns]
        lefts_neg = np.zeros(len(percent))
        for col in neg_cols:
            vals = percent[col].values
            bars = ax.barh(range(len(percent)), vals, left=lefts_neg, color=color_map.get(col), height=height, label=col, edgecolor="white")
            # anotação com contagem absoluta
            for i, v in enumerate(vals):
                cnt = counts.iloc[i].get(col, 0)
                if cnt:
                    x_pos = lefts_neg[i] + v / 2
                    ax.text(x_pos, i, f"{int(abs(cnt))}", ha="center", va="center", fontsize=8, color="black")
            lefts_neg += vals

        # positivo: Concordo, Concordo totalmente
        pos_cols = [c for c in ["Concordo", "Concordo totalmente"] if c in percent.columns]
        lefts_pos = np.zeros(len(percent))
        for col in pos_cols:
            vals = percent[col].values
            bars = ax.barh(range(len(percent)), vals, left=lefts_pos, color=color_map.get(col), height=height, label=col, edgecolor="white")
            for i, v in enumerate(vals):
                cnt = counts.iloc[i].get(col, 0)
                if cnt:
                    x_pos = lefts_pos[i] + v / 2
                    ax.text(x_pos, i, f"{int(abs(cnt))}", ha="center", va="center", fontsize=8, color="black")
            lefts_pos += vals

        # visual tweaks
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-100, 100)
        ax.set_yticks(range(len(percent)))
        ax.set_yticklabels(percent.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Porcentagem", fontsize=12)
        ax.set_title("Distribuição das respostas — Escala Likert", fontsize=13, weight="bold", pad=18)

        # legenda centralizada acima (forçar ordem específica)
        from matplotlib.patches import Patch
        desired = ["Discordo totalmente", "Discordo", "Neutro", "Concordo", "Concordo totalmente"]
        legend_labels = [lab for lab in desired if lab in percent.columns]
        handles = [Patch(facecolor=color_map[lab], edgecolor="white", label=lab) for lab in legend_labels]
        ax.legend(handles, legend_labels, title="", bbox_to_anchor=(0.5, 1.02), loc='upper center', ncol=len(legend_labels), frameon=False)

        plt.tight_layout()
        out_file = out_dir / "likert.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    # preparar retorno compacto (main.py espera dict com chaves usuais)
    # construir porcentagens por pergunta (mapping label -> (count, pct)) é complexo para multi-col;
    # mantemos counts e total para inspeção manual.
    return {
        "column": ", ".join(likert_cols),
        "counts": counts,
        "total": int(len(df)),
        "percentages": {},  # vazio por compatibilidade com main.py
        "image": image_path,
    }


if __name__ == "__main__":
    import os
    import sys
    import traceback

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)

    try:
        df = pd.read_csv(p, dtype=str, engine="python")
    except Exception as e:
        print("Falha ao ler CSV:", e)
        traceback.print_exc()
        raise SystemExit(1)

    res = analyze_likert(df, out_dir=project_root / "output")
    print({"column": res.get("column"), "total": res.get("total"), "image": res.get("image")})
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])
    else:
        print("Nenhuma imagem gerada. Verifique se matplotlib/numpy estão instalados e se há colunas Likert (cabeçalhos começando com '[').")