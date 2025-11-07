from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# matplotlib / numpy
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


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _map_choice_to_category(s: str) -> Optional[str]:
    """Mapear respostas livres para 'Obrigatório' ou 'Opcional' (ou None se não identificado)."""
    t = _normalize_text(s)
    if not t:
        return None
    if "obrig" in t:
        return "Obrigatório"
    if "opcion" in t or "opcional" in t:
        return "Opcional"
    # alguns formulários podem ter "Obrigatório / Opcional" juntos; priorizar mapeamento por palavra
    if "/" in t or "|" in t:
        if "obrig" in t and "opcion" in t:
            # tratar como ambos -> mapear como 'Obrigatório' por preferência, ou considerar "Opcional"?
            # Aqui contamos como 'Obrigatório' se contiver 'obrig' também.
            return "Obrigatório"
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


def analyze_obrig_optional(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Agrupa respostas em 'Obrigatório' / 'Opcional', gera gráfico de pizza com % + inteiro,
    usa laranja base #ff6002 em gradiente e salva em <project>/output/obrig_optional_pie.png por padrão.
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

    # detectar coluna provável
    col = None
    for c in df.columns:
        lc = str(c).lower()
        if "obrig" in lc and ("opcion" in lc or "opcional" in lc):
            col = c
            break
    if col is None:
        # tentativa permissiva
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in ("obrigatório", "obrigatorio", "obrig", "opcional", "opcion")):
                col = c
                break

    if col is None:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    # mapear valores para as duas categorias
    raw = df[col].astype(str).replace("", pd.NA).dropna().map(lambda x: x.strip())
    mapped = raw.map(lambda x: _map_choice_to_category(x))
    # filtrar não mapeados e contar
    mapped = mapped.dropna()
    if mapped.empty:
        return {"column": col, "counts": None, "total": 0, "percentages": {}, "image": None}

    counts = mapped.value_counts()  # Série com índice "Obrigatório"/"Opcional"
    total = int(counts.sum())
    percentages = {label: (int(cnt), float(cnt) / total * 100 if total else 0.0) for label, cnt in counts.items()}

    image_path = None
    if plt is None:
        # não pode plotar sem matplotlib
        return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": None}

    try:
        labels = counts.index.tolist()
        values = np.array([int(v) for v in counts.values], dtype=float)

        # gerar tons e ordenar para que o maior fique mais escuro
        shades = _orange_shades(BASE_ORANGE, len(values))  # light->dark
        # assign shades so that smallest -> lightest, largest -> darkest but keep label order
        order = np.argsort(values)  # indices of labels from smallest->largest
        color_map = [None] * len(values)
        for shade_idx, idx in enumerate(order):
            color_map[idx] = shades[shade_idx]

        # plot pie
        fig, ax = plt.subplots(figsize=(6, 6))
        total_float = float(values.sum()) if values.sum() > 0 else 1.0

        def autopct(pct):
            absolute = int(round(pct * total_float / 100.0))
            return f"{absolute}\n{pct:.1f}%"

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,  # legenda separada
            autopct=autopct,
            startangle=90,
            colors=color_map,
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
            textprops={"fontsize": 10},
        )
        ax.axis("equal")
        # legenda com apenas 'Obrigatório' / 'Opcional' + inteiro
        legend_labels = [f"{lab} ({int(v)})" for lab, v in zip(labels, values)]
        ax.legend(wedges, legend_labels, title="", loc="center left", bbox_to_anchor=(1.02, 0.5))

        ax.set_title("Obrigatório / Opcional")

        # ajustar cor do texto nas fatias para legibilidade
        for at, wg in zip(autotexts, wedges):
            face = wg.get_facecolor()
            luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            at.set_color("white" if luminance < 0.6 else "black")
            at.set_fontweight("bold")

        out_file = out_dir / "obrig_optional_pie.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    return {"column": col, "counts": counts, "total": total, "percentages": percentages, "image": image_path}


if __name__ == "__main__":
    import os
    import sys

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print(f"CSV não encontrado: {p}")
        raise SystemExit(1)
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_obrig_optional(df)
    print(res)
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])