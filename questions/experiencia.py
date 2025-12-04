from pathlib import Path
from typing import Dict, Any, Optional, List
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

# ordem fixa solicitada
EXP_ORDER: List[str] = [
    "Até um ano.",
    "Entre 1 e 2 anos.",
    "Entre 3 e 4 anos.",
    "Entre 5 e 6 anos.",
    "Entre 7 e 8 anos.",
    "Mais de 8 anos.",
    "Prefiro não responder",
]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _map_experiencia(v: str) -> Optional[str]:
    """Mapear resposta livre para uma das categorias em EXP_ORDER."""
    t = _normalize_text(v)
    if not t:
        return None
    # respostas explícitas
    if "prefiro" in t:
        return "Prefiro não responder"
    if "até um" in t or "até 1" in t or "<1" in t or "menos de 1" in t or "até um ano" in t:
        return "Até um ano."
    if ("entre" in t and ("1" in t and "2" in t)) or "1-2" in t or "1 a 2" in t or "1 a 2 anos" in t:
        return "Entre 1 e 2 anos."
    if ("entre" in t and ("3" in t and "4" in t)) or "3-4" in t or "3 a 4" in t:
        return "Entre 3 e 4 anos."
    if ("entre" in t and ("5" in t and "6" in t)) or "5-6" in t or "5 a 6" in t:
        return "Entre 5 e 6 anos."
    if ("entre" in t and ("7" in t and "8" in t)) or "7-8" in t or "7 a 8" in t:
        return "Entre 7 e 8 anos."
    if "mais de 8" in t or ">=8" in t or "8+" in t or "acima de 8" in t:
        return "Mais de 8 anos."

    # extrair primeiro número e inferir
    import re

    nums = re.findall(r"\d{1,3}", t)
    if nums:
        try:
            n = int(nums[0])
            if n <= 1:
                return "Até um ano."
            if 1 <= n <= 2:
                return "Entre 1 e 2 anos."
            if 3 <= n <= 4:
                return "Entre 3 e 4 anos."
            if 5 <= n <= 6:
                return "Entre 5 e 6 anos."
            if 7 <= n <= 8:
                return "Entre 7 e 8 anos."
            if n > 8:
                return "Mais de 8 anos."
        except Exception:
            pass

    # heurística simples por palavras-chave
    if "anos" in t and "mais" in t:
        return "Mais de 8 anos."
    return None


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
        t = i / max(1, n - 1)  # 0..1
        mix = 0.35 + 0.65 * t
        rgb = white * (1.0 - mix) + base_rgb * mix
        shades.append(tuple(rgb.tolist()))
    return shades


def find_column(df: pd.DataFrame) -> Optional[str]:
    """Detecta coluna da pergunta de experiência por heurística."""
    # preferência por nome exato
    exact = "Qual seu tempo total de experiência na engenharia de software?"
    if exact in df.columns:
        return exact
    # procurar por palavras-chave
    for c in df.columns:
        lc = str(c).lower()
        if "tempo" in lc and "experien" in lc:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "experien" in lc or ("tempo" in lc and "anos" in lc):
            return c
    return None


def analyze_experiencia(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Agrupa respostas da pergunta de experiência nas categorias definidas em EXP_ORDER,
    gera gráfico de barras horizontal com gradiente laranja (base #ff6002) e salva em output.
    Retorna dict similar aos outros analisadores.
    """
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    col = find_column(df)
    if col is None:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    raw = df[col].astype(str).replace("", pd.NA).dropna().map(lambda x: x.strip())
    mapped = raw.map(lambda x: _map_experiencia(x))
    mapped = mapped.dropna()
    if mapped.empty:
        return {"column": col, "counts": None, "total": 0, "percentages": {}, "image": None}

    counts = mapped.value_counts()
    counts = counts.reindex(EXP_ORDER, fill_value=0)
    total = int(counts.sum())
    percentages = {lbl: (int(cnt), float(cnt) / total * 100 if total else 0.0) for lbl, cnt in counts.items()}

    image_path = None
    if plt is None:
        return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": None}

    try:
        labels = counts.index.tolist()
        values = counts.values.astype(float)

        # gerar tons (light->dark) e mapear para que maior fique mais escuro
        shades = _orange_shades(BASE_ORANGE, len(values))
        if np is not None:
            order = np.argsort(values)
        else:
            order = list(range(len(values)))
        color_map = [None] * len(values)
        for shade_idx, idx in enumerate(order):
            color_map[idx] = shades[shade_idx]

        # plot horizontal
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=color_map, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Contagem")
        # ax.set_title("Tempo total de experiência na engenharia de software")

        # anotar números dentro da barra quando couber
        vmax = values.max() if len(values) else 1
        for bar, val in zip(bars, values):
            pct = (val / total * 100) if total else 0.0
            label = f"{pct:.1f}% ({int(val)})"
            inside_threshold = vmax * 0.15
            if val >= inside_threshold:
                x = val - vmax * 0.02
                ha = "right"
            else:
                x = val + vmax * 0.02
                ha = "left"
            try:
                face = bar.get_facecolor()
                luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
                text_color = "white" if luminance < 0.6 else "black"
            except Exception:
                text_color = "black"
            ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, color=text_color, fontsize=12)

        plt.tight_layout()
        out_file = out_dir / "experiencia_barras_horizontais.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": image_path}


# execução direta para debugging / geração de imagem
if __name__ == "__main__":
    import os
    import sys

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_experiencia(df)
    print("analyze_experiencia retornou:", {"column": res["column"], "total": res["total"], "image": res["image"]})
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])