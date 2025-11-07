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

EMP_ORDER: List[str] = [
    "Até 9 funcionários (Microempresa)",
    "De 10 a 49 funcionários (Pequena empresa)",
    "De 50 a 99 funcionários (Média empresa)",
    "100 ou mais funcionários (Grande empresa)",
    "Prefiro não responder",
]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _orange_shades(base_hex: str, n: int) -> list:
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


def _map_empresa(v: str) -> str:
    t = _normalize_text(v)
    if not t:
        return "Outros"
    if "prefiro" in t:
        return "Prefiro não responder"
    if any(k in t for k in ("até 9", "até 9 funcionários", "micro", "microempresa", "<=9", "até 9 funcionários (microempresa)")):
        return "Até 9 funcionários (Microempresa)"
    if any(k in t for k in ("10 a 49", "10-49", "10 a 49 funcionários", "pequena", "pequena empresa")):
        return "De 10 a 49 funcionários (Pequena empresa)"
    if any(k in t for k in ("50 a 99", "50-99", "50 a 99 funcionários", "média", "media empresa", "média empresa")):
        return "De 50 a 99 funcionários (Média empresa)"
    if any(k in t for k in ("100", "100 ou mais", "100 ou mais funcionários", ">=100", "mais de 100", "grande", "grande empresa")):
        return "100 ou mais funcionários (Grande empresa)"
    # try digits
    import re
    nums = re.findall(r"\d+", t)
    if nums:
        try:
            n = int(nums[0])
            if n <= 9:
                return "Até 9 funcionários (Microempresa)"
            if 10 <= n <= 49:
                return "De 10 a 49 funcionários (Pequena empresa)"
            if 50 <= n <= 99:
                return "De 50 a 99 funcionários (Média empresa)"
            if n >= 100:
                return "100 ou mais funcionários (Grande empresa)"
        except Exception:
            pass
    return "Outros"


def find_column(df: pd.DataFrame) -> Optional[str]:
    exact = "Quantos funcionários a empresa possui?"
    if exact in df.columns:
        return exact
    for c in df.columns:
        lc = str(c).lower()
        if "funcionários" in lc or "quantos funcionários" in lc or "número de funcionários" in lc:
            return c
    return None


def analyze_empresa(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analisa a pergunta sobre número de funcionários, agrupa nas categorias definidas em EMP_ORDER,
    gera gráfico de barras horizontal com gradiente de laranja (base #ff6002) e salva em output.
    Retorna dict: { column, counts (Series), total, percentages (dict), image (Path|None) }.
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

    s = df[col].astype(str).replace("", pd.NA).dropna().map(lambda x: x.strip())
    mapped = s.map(lambda x: _map_empresa(x))
    if mapped.empty:
        return {"column": col, "counts": None, "total": 0, "percentages": {}, "image": None}

    counts = mapped.value_counts()
    counts = counts.reindex(EMP_ORDER, fill_value=0)
    total = int(counts.sum())
    percentages = {lbl: (int(cnt), float(cnt) / total * 100 if total else 0.0) for lbl, cnt in counts.items()}

    image_path = None
    if plt is None:
        return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": None}

    try:
        labels = counts.index.tolist()
        values = counts.values.astype(float)

        # generate shades and map so largest -> darkest
        shades = _orange_shades(BASE_ORANGE, len(values))
        if np is not None:
            order = np.argsort(values)
        else:
            order = list(range(len(values)))
        color_map = [None] * len(values)
        for shade_idx, idx in enumerate(order):
            color_map[idx] = shades[shade_idx]

        # plot horizontal bars
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=color_map, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Contagem")
        ax.set_title("Quantos funcionários a empresa possui?")

        # annotate numbers inside bars when possible
        vmax = values.max() if len(values) else 1
        for bar, val in zip(bars, values):
            pct = (val / total * 100) if total else 0.0
            label = f"{int(val)} ({pct:.1f}%)"
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
            ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, color=text_color, fontsize=9)

        plt.tight_layout()
        out_file = out_dir / "empresa_barras_horizontais.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": image_path}


if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_empresa(df)
    print({"column": res["column"], "total": res["total"], "image": res["image"]})