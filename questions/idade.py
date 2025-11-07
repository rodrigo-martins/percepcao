from pathlib import Path
from typing import Optional, Dict, Any
import re

import pandas as pd


def find_column(df: pd.DataFrame) -> Optional[str]:
    """Encontra a coluna da pergunta 'Qual é a sua idade?' por nome ou por valores contendo idades."""
    for col in df.columns:
        name = str(col).lower()
        if "idade" in name or "anos" in name:
            return col
    # procurar por valores que contenham números plausíveis de idade
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if s.str.contains(r"\b\d{1,3}\b").any():
            # contar quantos valores plausíveis (10-120) aparecem
            nums = s.str.extractall(r"(\d{1,3})")[0].astype(int)
            if not nums.empty and nums.between(10, 120).sum() > 0:
                return col
    return None


def _parse_age_value(v: str) -> Optional[float]:
    """Extrai número(s) de uma string de idade. Se range '25-34' retorna média, senão primeiro número."""
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    # extrai todos os números
    nums = re.findall(r"\d{1,3}", v)
    if not nums:
        return None
    nums = [int(n) for n in nums]
    # se for um range com dois números, usa a média
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        return float((a + b) / 2)
    return float(nums[0])


def summarize_ages(df: pd.DataFrame, q_col: str) -> pd.Series:
    """Retorna uma Series com idades (valores numéricos)."""
    s = df[q_col].dropna().astype(str)
    ages = s.map(_parse_age_value).dropna().astype(float)
    return ages


def plot_ages(ages: pd.Series, out_dir: Path) -> Optional[Path]:
    """
    Gera gráfico de barras das contagens por idade, sobrepõe curva normal (média/std),
    adiciona linhas tracejadas em Q1, Q2(mediana) e Q3, inclui caixa com describe() e salva imagem.
    Retorna Path da imagem ou None se matplotlib ausente ou dados insuficientes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    if ages.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # contagens por idade inteira (arredonda para inteiro para barras)
    ages_int = ages.round().astype(int)
    counts = ages_int.value_counts().sort_index()
    x = counts.index.values
    y = counts.values

    mean = float(ages.mean())
    std = float(ages.std(ddof=0)) if float(ages.std(ddof=0)) > 0 else 1.0
    q1 = float(ages.quantile(0.25))
    q2 = float(ages.quantile(0.50))  # mediana
    q3 = float(ages.quantile(0.75))
    total = len(ages)

    fig, ax = plt.subplots(figsize=(9, 5))
    # barras laranja
    ax.bar(x, y, width=0.8, color="#ff6002", label="Contagem por idade")

    # linha da distribuição normal (preta) escalada para counts
    xs = np.linspace(max(x.min() - 2, 0), x.max() + 2, 300)
    pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
    pdf_scaled = pdf * total
    ax.plot(xs, pdf_scaled, color="black", lw=2)  # sem label para manter apenas um label no gráfico

    # linhas tracejadas em Q1, Q2 (mediana) e Q3 sem adicionar entradas na legenda
    ax.axvline(q1, color="black", linestyle="--", lw=1)
    ax.axvline(q2, color="black", linestyle="--", lw=1)
    ax.axvline(q3, color="black", linestyle="--", lw=1)
    # anotações próximas ao topo para cada quartil
    ylim_top = max(y.max(), pdf_scaled.max()) * 0.98
    ax.text(q1, ylim_top, f"Q1={q1:.1f}", ha="center", va="top", fontsize=9)
    ax.text(q2, ylim_top, f"Q2={q2:.1f}", ha="center", va="top", fontsize=9)
    ax.text(q3, ylim_top, f"Q3={q3:.1f}", ha="center", va="top", fontsize=9)

    ax.set_xlabel("Idade")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuição de Idade com curva normal")
    # somente um label na legenda (das barras)
    ax.legend(loc="upper left")

    # caixa com describe() em formato legível
    desc = ages.describe()
    stats_lines = [
        f"count: {int(desc['count'])}",
        f"mean: {desc['mean']:.2f}",
        f"std: {desc['std']:.2f}",
        f"min: {desc['min']:.1f}",
        f"25%: {desc['25%']:.1f}",
        f"50%: {desc['50%']:.1f}",
        f"75%: {desc['75%']:.1f}",
        f"max: {desc['max']:.1f}",
    ]
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    out_path = out_dir / "idade.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analyze_idade(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Função única para análise da pergunta 'Qual é a sua idade?'
    Retorna dicionário com:
      - column: nome da coluna encontrada (ou None)
      - ages: pandas.Series numérica das idades (ou None)
      - counts: pandas.Series contagens por idade (ou None)
      - total: int número de respostas válidas
      - describe: pandas.Series retornado por ages.describe() (ou None)
      - image: Path da imagem gerada (ou None)
    """
    out_dir = Path(out_dir) if out_dir is not None else None
    col = find_column(df)
    if col is None:
        return {"column": None, "ages": None, "counts": None, "total": 0, "describe": None, "image": None}

    ages = summarize_ages(df, col)
    if ages.empty:
        return {"column": col, "ages": ages, "counts": None, "total": 0, "describe": None, "image": None}

    counts = ages.round().astype(int).value_counts().sort_index()
    total = int(len(ages))
    describe = ages.describe()

    image = None
    if out_dir:
        image = plot_ages(ages, out_dir)

    return {
        "column": col,
        "ages": ages,
        "counts": counts,
        "total": total,
        "describe": describe,
        "image": image,
    }


# novo: permitir execução direta para gerar/save gráfico em <project>/output
if __name__ == "__main__":
    import os
    import sys

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print(f"CSV não encontrado: {p}")
        raise SystemExit(1)

    try:
        df = pd.read_csv(p, dtype=str, engine="python")
    except Exception as e:
        print("Falha ao ler CSV:", e)
        raise SystemExit(1)

    out_dir = project_root / "output"
    res = analyze_idade(df, out_dir=out_dir)
    print("analyze_idade retornou:")
    print({k: (v if k != "ages" else f"<Series len={len(v)}>" ) for k, v in res.items()})
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])
    else:
        print("Nenhuma imagem gerada. Verifique se matplotlib está instalado.")