from pathlib import Path
from typing import Optional, Dict, Any
import re
import unicodedata

import pandas as pd

try:
    import numpy as np
    import matplotlib.colors as mcolors
except Exception:
    np = None
    mcolors = None

BASE_ORANGE = "#ff6002"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_text(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^\w\s-]", "", s)
    return s


def _map_to_three(v: str) -> str:
    """
    Mapear qualquer resposta para uma das três categorias:
    'Feminino', 'Masculino', 'Prefiro não responder'.
    Valores não reconhecidos caem em 'Prefiro não responder'.
    """
    s = _normalize_text(v)
    if s == "" or s in {"nan", "none"}:
        
        return "Prefiro não responder"
    # Masculino
    if any(tok in s for tok in ("mascul", "homem", "masculino", "mas", " m ")):
        return "Masculino"
    # Feminino
    if any(tok in s for tok in ("femin", "mulher", "feminino", "fem", " f ")):
        return "Feminino"
    # variações explícitas de "prefiro não responder"
    if any(tok in s for tok in ("prefiro", "nao responder", "nao dizer", "nao-responder", "pref nao", "prefiro nao")):
        return "Prefiro não responder"
    # fallback para garantir apenas as três categorias
    return "Prefiro não responder"


def find_column(df: pd.DataFrame) -> Optional[str]:
    """Preferir colunas cujo nome indique gênero; caso contrário, escolher por score de tokens."""
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ("gênero", "genero", "sexo", "gender")):
            return col
    # pontuar por ocorrência de tokens
    keywords = ("mascul", "homem", "femin", "mulher", "prefiro", "nao", "não")
    best_col = None
    best_score = 0.0
    for col in df.columns:
        s = df[col].dropna().astype(str).map(_normalize_text)
        if s.empty:
            continue
        score = s.apply(lambda x: any(k in x for k in keywords)).mean()
        if score > best_score:
            best_score = score
            best_col = col
    if best_score >= 0.02:
        return best_col
    return None


def summarize_gender(df: pd.DataFrame, q_col: str) -> pd.Series:
    """Retorna contagens nas três categorias, sempre na mesma ordem."""
    s = df[q_col].astype(str).replace("", pd.NA).dropna().map(_map_to_three)
    labels = [ "Masculino", "Feminino", "Prefiro não responder"]
    counts = s.value_counts()
    data = {lbl: int(counts.get(lbl, 0)) for lbl in labels}
    return pd.Series(data)


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


def plot_pie(counts: pd.Series, out_dir: Path) -> Optional[Path]:
    """Gera gráfico de pizza em escala de laranja; mostra porcentagem + número absoluto em cada fatia.
    Legenda na parte superior. Cores graduais: menor -> mais claro, maior -> laranja mais escuro.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # garantir que as três categorias existam na ordem desejada
    labels = ["Masculino", "Feminino", "Prefiro não responder"]
    sizes = np.array([float(counts.get(lbl, 0)) for lbl in labels])
    total = sizes.sum() if sizes is not None and sizes.sum() > 0 else 1.0

    # gerar cores a partir do BASE_ORANGE (menor->mais claro, maior->mais escuro)
    if np is None or mcolors is None:
        # fallback: usar cor base repetida
        colors = [BASE_ORANGE] * len(labels)
    else:
        shades = _orange_shades(BASE_ORANGE, len(labels))  # light -> dark
        order = np.argsort(sizes)  # indices from smallest->largest
        color_map = [None] * len(labels)
        for shade_idx, idx in enumerate(order):
            color_map[idx] = shades[shade_idx]
        colors = color_map

    explode = [0.05] * len(sizes)

    def autopct_fn(pct):
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({absolute})"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=autopct_fn,
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        textprops={"fontsize": 20, "fontweight": "bold"},
    )
    ax.axis("equal")
    # legenda na parte superior (SEM inteiro)
    legend_labels = [f"{lab}" for lab in labels]
    ax.legend(wedges, legend_labels, title="", loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=20)

    # ajustar cor dos textos dentro das fatias para garantir legibilidade
    for autotext, wedge in zip(autotexts, wedges):
        autotext.set_fontsize(20)
        autotext.set_fontweight("normal")
        face = wedge.get_facecolor()  # (r,g,b,a)
        luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        autotext.set_color("white" if luminance < 0.6 else "black")

    out_path = out_dir / "genero_pie.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_genero(df: pd.DataFrame, out_dir: Optional[Path] = None, debug: bool = False) -> Dict[str, Any]:
    # padrão: salvar em <project_root>/output se out_dir não for fornecido
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    col = find_column(df)
    counts = summarize_gender(df, col)
    total = int(counts.sum())
    percentages = counts / total * 100 if total > 0 else counts * 0

    # garantir que retornamos o path da imagem (se plot_pie for usado)
    image = None
    try:
        image = plot_pie(counts, out_dir)  # plot_pie definida acima
    except Exception:
        if debug:
            import traceback
            traceback.print_exc()

    return {"column": col, "counts": counts, "total": int(counts.sum()), "percentages": percentages, "image": image}

# permitir execução direta: python -m questions.genero  ou  python questions/genero.py
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
    res = analyze_genero(df, out_dir=project_root / "output", debug=True)
    print("analyze_genero retornou:", {k: (v if k != "counts" else v.to_dict()) for k, v in res.items()})
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])
