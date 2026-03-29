import argparse
from pathlib import Path
from itertools import combinations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# --- Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "analise_tematica_qq1.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Mapeamento de temas (CSV → ID) ordenado por frequência ---
THEME_LABELS = {
    "Curso e Treinamentos": "T1",
    "Aprendizado prático e aplicado": "T2",
    "Educação Formal": "T3",
    "Aprendizado social e networking": "T4",
    "Desenvolvimento de liderança e soft skills": "T5",
}

# Temas excluídos do grafo (observação emergente)
EXCLUDED_THEMES = {"Aprendizado autodirigido"}

THEME_COLORS = {
    "T1": "#E74C3C",
    "T2": "#F39C12",
    "T3": "#3498DB",
    "T4": "#2ECC71",
    "T5": "#1ABC9C",
}

# --- Textos por idioma ---
TRANSLATIONS = {
    "pt": {
        "node_labels": {
            "T1": "T1\nAtualização\nTécnica",
            "T2": "T2\nAprendizado\nPrático",
            "T3": "T3\nEducação\nFormal",
            "T4": "T4\nAprendizado\nSocial",
            "T5": "T5\nLiderança e\nSoft Skills",
        },
        "output_file": "grafo_coocorrencia_temas_qq1.png",
    },
    "en": {
        "node_labels": {
            "T1": "T1\nContinuous\nTechnical\nUpdating",
            "T2": "T2\nPractical\nLearning",
            "T3": "T3\nFormal\nEducation",
            "T4": "T4\nSocial\nLearning",
            "T5": "T5\nLeadership &\nSoft Skills",
        },
        "output_file": "grafo_coocorrencia_temas_qq1_en.png",
    },
}


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def compute_frequencies(df: pd.DataFrame):
    # Temas estão nas colunas 4, 5 e 6 (Etapa 3 e adjacentes)
    theme_cols = [df.columns[4], df.columns[5], df.columns[6]]
    freq = {}
    cooccurrence = {}

    for _, row in df.iterrows():
        themes_in_row = set()
        for col in theme_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                raw = str(val).strip()
                if raw in EXCLUDED_THEMES:
                    continue
                label = THEME_LABELS.get(raw)
                if label:
                    themes_in_row.add(label)

        for t in themes_in_row:
            freq[t] = freq.get(t, 0) + 1

        for pair in combinations(sorted(themes_in_row), 2):
            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

    return freq, cooccurrence


def build_graph(freq, cooccurrence):
    G = nx.Graph()
    for node, count in freq.items():
        G.add_node(node, frequency=count)
    for (u, v), weight in cooccurrence.items():
        G.add_edge(u, v, weight=weight)
    return G


def draw_graph(G, freq, cooccurrence, lang: str):
    t = TRANSLATIONS[lang]
    node_labels = t["node_labels"]
    output_file = t["output_file"]

    fig, ax = plt.subplots(figsize=(18, 13))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Layout fixo para 5 nós (espaçados)
    pos = {
        "T1": (0.0, 0.0),
        "T2": (-1.2, -1.2),
        "T3": (1.2, 1.2),
        "T4": (1.4, -0.7),
        "T5": (-1.2, 1.1),
    }

    # Tamanho dos nós proporcional à frequência
    max_freq = max(freq.values())
    min_size, max_size = 25000, 55000
    node_sizes = [
        min_size + (max_size - min_size) * (freq.get(node, 1) / max_freq)
        for node in G.nodes()
    ]

    node_colors = [THEME_COLORS[node] for node in G.nodes()]

    # Arestas
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [3.5 + 18.0 * (w / max_weight) for w in edge_weights]
    edge_alphas = [0.3 + 0.5 * (w / max_weight) for w in edge_weights]

    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=width, alpha=alpha,
            edge_color="#555555", ax=ax,
        )

    edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=22, font_weight="bold", font_color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=node_colors, edgecolors="#333333",
        linewidths=1.5, alpha=0.92, ax=ax,
    )

    # Rótulos dentro dos nós
    for node, (x, y) in pos.items():
        f = freq.get(node, 0)
        label = f"{node_labels[node]}\n(n={f})"
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=20, fontweight="bold", color="white",
            linespacing=1.15,
        )

    ax.set_xlim(-1.7, 1.9)
    ax.set_ylim(-1.7, 1.7)
    ax.axis("off")

    output_path = OUTPUT_DIR / output_file
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.5, facecolor="white")
    plt.close(fig)
    print(f"Grafo salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grafo de coocorrência de temas QQ1")
    parser.add_argument(
        "--lang", choices=["pt", "en"], default="pt",
        help="Idioma dos rótulos: pt (padrão) ou en",
    )
    args = parser.parse_args()

    df = load_data()
    freq, cooccurrence = compute_frequencies(df)
    G = build_graph(freq, cooccurrence)
    draw_graph(G, freq, cooccurrence, lang=args.lang)


if __name__ == "__main__":
    main()
