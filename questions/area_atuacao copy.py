from pathlib import Path
from typing import Dict, Any, Optional
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


def find_column(df: pd.DataFrame) -> Optional[str]:
    exact = "Qual é a sua principal área de atuação na engenharia de software no momento?"
    if exact in df.columns:
        return exact
    for c in df.columns:
        lc = str(c).lower()
        if "área de atuação" in lc or "area de atuacao" in lc:
            return c
    return None


def analyze_area_atuacao(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent

    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    col = find_column(df)
    if col is None:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    # 👉 respostas já padronizadas → apenas agrupar
    s = df[col].dropna().astype(str).str.strip()
    counts = s.value_counts()
    total = int(counts.sum())

    percentages = {
        k: (int(v), v / total * 100 if total else 0.0)
        for k, v in counts.items()
    }

    image_path = None
    if plt is None:
        return {
            "column": col,
            "counts": counts,
            "total": total,
            "percentages": percentages,
            "image": None,
        }

    try:
        labels = counts.index.tolist()
        values = counts.values.astype(float)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))))
        bars = ax.barh(labels, values, color=BASE_ORANGE)

        ax.invert_yaxis()
        ax.set_xlabel("Contagem", fontsize=14, fontweight="bold")

        vmax = values.max() if len(values) else 1
        for bar, val in zip(bars, values):
            pct = val / total * 100 if total else 0
            ax.text(
                val + vmax * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({int(val)})",
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_xlim(0, vmax * 1.25)
        plt.tight_layout()

        out_file = out_dir / "area_atuacao_barras_horizontais.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        image_path = out_file

    except Exception:
        image_path = None

    return {
        "column": col,
        "counts": counts,
        "total": total,
        "percentages": percentages,
        "image": image_path,
    }


if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "tratado.csv"))
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    res = analyze_area_atuacao(df)

    print({
        "column": res["column"],
        "total": res["total"],
        "image": res["image"],
    })
