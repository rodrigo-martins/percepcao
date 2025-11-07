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

# ordem obrigatória pedida pelo usuário
INSTR_ORDER: List[str] = [
    "Ensino Médio incompleto",
    "Ensino Médio completo",
    "Ensino Superior incompleto",
    "Ensino Superior completo",
    "Pós-graduação incompleta",
    "Pós-graduação completa",
]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _map_instrução(v: str) -> Optional[str]:
    t = _normalize_text(v)
    if not t:
        return None
    # mapeamentos simples/heurísticos
    if "ensino médio incompleto" in t or "ensino medio incompleto" in t:
        return "Ensino Médio incompleto"
    if "ensino médio completo" in t or "ensino medio completo" in t:
        return "Ensino Médio completo"
    if "ensino superior incompleto" in t or "ensino superior (incompleto)" in t:
        return "Ensino Superior incompleto"
    if "ensino superior completo" in t or "ensino superior completo" in t:
        return "Ensino Superior completo"
    if "pós-graduação incompleta" in t or "pos-graduação incompleta" in t or "pós graduação incompleta" in t or "pos graduacao incompleta" in t:
        return "Pós-graduação incompleta"
    if "pós-graduação completa" in t or "pos-graduação completa" in t or "pós graduacao completa" in t:
        return "Pós-graduação completa"
    # variações curtas
    if "pós" in t and "incomplet" in t:
        return "Pós-graduação incompleta"
    if "pós" in t or "pos" in t or "especialização" in t or "mestrado" in t or "doutorado" in t:
        # prefer completa when ambiguous? fallback to completa
        return "Pós-graduação completa"
    # fallback: try to match keywords
    if "ensino médio" in t:
        return "Ensino Médio completo"
    if "ensino superior" in t or "bacharel" in t or "licenciatura" in t:
        return "Ensino Superior completo"
    return None


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


def analyze_instrucao(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Agrupa respostas da pergunta 'mais alto grau de instrução' usando ordem fixa e gera
    um gráfico de barras horizontal com gradiente de laranja (base #ff6002).
    Retorna dict com keys: column, counts (Series), total, percentages, image (Path|None).
    """
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # encontrar coluna por heurística
    candidate = None
    for c in df.columns:
        lc = str(c).lower()
        if "grau de instrução" in lc or "grau de instru" in lc or "grau de instrucao" in lc or "mais alto grau" in lc:
            candidate = c
            break
    if candidate is None:
        # tentativa permissiva por palavras
        for c in df.columns:
            lc = str(c).lower()
            if any(k in lc for k in ("instrução", "instrucao", "grau", "mais alto grau")):
                candidate = c
                break

    if candidate is None:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    # mapear e contar
    s = df[candidate].astype(str).replace("", pd.NA).dropna().map(lambda x: x.strip())
    mapped = s.map(lambda x: _map_instrução(x))
    mapped = mapped.dropna()
    if mapped.empty:
        return {"column": candidate, "counts": None, "total": 0, "percentages": {}, "image": None}

    counts = mapped.value_counts()
    # garantir todas as categorias da ordem existam (mesmo que zero)
    counts = counts.reindex(INSTR_ORDER, fill_value=0)
    total = int(counts.sum())
    percentages = {lbl: (int(cnt), float(cnt) / total * 100 if total else 0.0) for lbl, cnt in counts.items()}

    image_path = None
    if plt is None:
        return {"column": candidate, "counts": counts, "total": total, "percentages": percentages, "image": None}

    try:
        labels = counts.index.tolist()
        values = counts.values.astype(float)

        # gerar tons de laranja (light -> dark) e mapear para valores: maior -> mais escuro
        shades = _orange_shades(BASE_ORANGE, len(values))  # light->dark
        # order indices from smallest->largest
        if np is not None:
            order = np.argsort(values)
        else:
            order = list(range(len(values)))
        color_map = [None] * len(values)
        for shade_idx, idx in enumerate(order):
            color_map[idx] = shades[shade_idx]

        # criar gráfico horizontal
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=color_map, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # maior em cima (visual preferencial)
        ax.set_xlabel("Contagem")
        ax.set_title("Grau de Instrução")

        # anotar contagem e porcentagem dentro da barra quando couber
        vmax = values.max() if len(values) else 1
        for i, (bar, val) in enumerate(zip(bars, values)):
            pct = (val / total * 100) if total else 0.0
            label = f"{int(val)} ({pct:.1f}%)"
            # posição dentro quando a barra for suficientemente larga, caso contrário fora à direita
            inside_threshold = vmax * 0.15
            if val >= inside_threshold:
                x = val - vmax * 0.02
                ha = "right"
            else:
                x = val + vmax * 0.02
                ha = "left"

            # escolher cor do texto com base na cor da barra para garantir contraste
            try:
                face = bar.get_facecolor()  # RGBA
                luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
                text_color = "white" if luminance < 0.6 else "black"
            except Exception:
                text_color = "black"

            ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, color=text_color, fontsize=9)

        plt.tight_layout()
        out_file = out_dir / "instrucao_barras_horizontais.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    return {"column": candidate, "counts": counts, "total": total, "percentages": percentages, "image": image_path}


# execução direta para debugging
if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_instrucao(df)
    print("analyze_instrucao retornou:", {"column": res["column"], "total": res["total"], "image": res["image"]})