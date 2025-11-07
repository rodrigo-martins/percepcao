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

# categorias solicitadas (adicionei "Suporte")
AREA_ORDER: List[str] = [
    "Desenvolvimento",
    "Arquitetura",
    "DevOps",
    "Design (UI/UX)",
    "Liderança Técnica",
    "Gestão de Pessoas / Projetos",
    "Ciência de Dados",
    "Controle de Qualidade (QA)",
    "Suporte",
    "Outros",
]


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _map_area(v: str) -> str:
    t = _normalize_text(v)
    if not t:
        return "Outros"
    # Suporte (devemos mapear antes de termos genéricos como "outros")
    if any(k in t for k in (
        "analista de suporte", "suporte técnico", "suporte", "suporte técnico de ti",
        "analista de suporte junior", "analista de suporte júnior",
        "na área do suporte", "não trabalho com engenharia de software. sou analista de suporte"
    )):
        return "Suporte"
    # Desenvolvimento
    if any(k in t for k in ("desenvolv", "developer", "dev ")):
        return "Desenvolvimento"
    if "arquitet" in t:
        return "Arquitetura"
    # DevOps e infraestrutura / implantação
    if any(k in t for k in (
        "devops", "analista de implantação", "analista de implantacao",
        "analista de infraestrutura", "analista de infraestrutura de ti", "infraestrutura", "infra"
    )):
        return "DevOps"
    if "ui/ux" in t or "ui ux" in t or "design" in t:
        return "Design (UI/UX)"
    if "liderança técnica" in t or "lideranca tecnica" in t or "technical lead" in t or "liderança" in t or "lead" in t:
        return "Liderança Técnica"
    if any(k in t for k in ("gestão", "gestao", "project", "projetos", "people", "gerente", "coordena")):
        return "Gestão de Pessoas / Projetos"
    # Ciência de Dados: incluir Engenharia de Dados, Dados, BI
    if any(k in t for k in ("engenharia de dados", "engenheiro de dados", "engenharia de dados", "dados", "bi", "business intelligence")):
        return "Ciência de Dados"
    if any(k in t for k in ("qualidade", "qa", "teste", "testes", "controle de qualidade")):
        return "Controle de Qualidade (QA)"
    # fallback: if response matches any label substring
    for lbl in AREA_ORDER:
        if lbl.lower() in t:
            return lbl
    return "Outros"


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


def find_column(df: pd.DataFrame) -> Optional[str]:
    exact = "Qual é a sua principal área de atuação na engenharia de software no momento?"
    if exact in df.columns:
        return exact
    for c in df.columns:
        lc = str(c).lower()
        if "área de atuação" in lc or "area de atuacao" in lc or ("área" in lc and "atuação" in lc) or ("área" in lc and "atuação" not in lc and "atuacao" in lc):
            return c
    # fallback por palavras-chave
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ("área", "area", "atuacao", "atuação", "atuação na engenharia", "area de atuação")):
            return c
    return None


def analyze_area_atuacao(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
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
    mapped = s.map(lambda x: _map_area(x))
    mapped = mapped.dropna()
    if mapped.empty:
        return {"column": col, "counts": None, "total": 0, "percentages": {}, "image": None}

    counts = mapped.value_counts()
    # garantir categorias fixas + Outros
    counts = counts.reindex(AREA_ORDER, fill_value=0)
    # ordenar pelo maior número (desc)
    counts_sorted = counts.sort_values(ascending=False)
    total = int(counts_sorted.sum())
    percentages = {lbl: (int(cnt), float(cnt) / total * 100 if total else 0.0) for lbl, cnt in counts_sorted.items()}

    image_path = None
    if plt is None:
        return {"column": col, "counts": counts_sorted, "total": total, "percentages": percentages, "image": None}

    try:
        labels = counts_sorted.index.tolist()
        values = counts_sorted.values.astype(float)
        n = len(labels)
        shades = _orange_shades(BASE_ORANGE, n)  # light->dark
        # we want largest -> darkest; counts_sorted is desc, so reverse shades
        colors = list(reversed(shades))

        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * n)))
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Contagem")
        ax.set_title("Principal área de atuação (engenharia de software)")

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
        out_file = out_dir / "area_atuacao_barras_horizontais.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    return {"column": col, "counts": counts_sorted, "total": total, "percentages": percentages, "image": image_path}


if __name__ == "__main__":
    import os, sys
    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)
    df = pd.read_csv(p, dtype=str, engine="python")
    res = analyze_area_atuacao(df)

    print({"column": res["column"], "total": res["total"], "image": res["image"]})

    # imprimir informação da categoria "Outros" e exemplos de respostas originais mapeadas como 'Outros'
    outros_count = int(res["counts"].get("Outros", 0)) if res.get("counts") is not None else 0
    print(f"'Outros' — contagem: {outros_count}")

    if outros_count > 0:
        # mostrar até 20 respostas originais que foram classificadas como "Outros"
        col = res["column"]
        original = df[col].astype(str).replace("", pd.NA).dropna().map(lambda x: x.strip())
        mapped = original.map(lambda x: _map_area(x))
        others = original[mapped == "Outros"]
        examples = others.value_counts().head(20)
        print("Exemplos de respostas mapeadas como 'Outros' (top 20):")
        print(examples.to_string())