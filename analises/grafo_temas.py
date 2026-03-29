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
DATA_PATH = BASE_DIR / "data" / "analise_tematica_qq2_v2.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Mapeamento de temas (CSV → ID) ---
THEME_LABELS = {
    "Aplicabilidade prática e alinhamento às necessidades": "T1",
    "Qualidade pedagogica didatica e organizacao": "T2",
    "Tempo e condicoes estruturais para aprendizagem": "T3",
    "Incentivos reconhecimento e apoio institucional": "T4",
    "Interacao mentoria e troca social": "T5",
}

THEME_COLORS = {
    "T1": "#E74C3C",
    "T2": "#3498DB",
    "T3": "#2ECC71",
    "T4": "#F39C12",
    "T5": "#9B59B6",
}

# --- Textos por idioma ---
TRANSLATIONS = {
    "pt": {
        "node_labels": {
            "T1": "T1\nAplicabilidade",
            "T2": "T2\nQualidade\nPedagógica",
            "T3": "T3\nTempo",
            "T4": "T4\nIncentivos",
            "T5": "T5\nInteração\nSocial",
        },
        "output_file": "grafo_coocorrencia_temas.png",
    },
    "en": {
        "node_labels": {
            "T1": "T1\nApplicability",
            "T2": "T2\nPedagogical\nQuality",
            "T3": "T3\nTime",
            "T4": "T4\nIncentives",
            "T5": "T5\nSocial\nInteraction",
        },
        "output_file": "grafo_coocorrencia_temas_en.png",
    },
}


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def compute_frequencies(df: pd.DataFrame):
    theme_cols = ["Tema_1", "Tema_2", "Tema_3"]
    freq = {}
    cooccurrence = {}

    for _, row in df.iterrows():
        themes_in_row = set()
        for col in theme_cols:
            val = row.get(col)
            if pd.notna(val) and val.strip():
                label = THEME_LABELS.get(val.strip())
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

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Tamanho dos nós proporcional à frequência
    max_freq = max(freq.values())
    min_size, max_size = 3000, 7000
    node_sizes = [
        min_size + (max_size - min_size) * (freq.get(node, 1) / max_freq)
        for node in G.nodes()
    ]

    node_colors = [THEME_COLORS[node] for node in G.nodes()]

    # Arestas
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1.5 + 8.0 * (w / max_weight) for w in edge_weights]
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
        font_size=11, font_weight="bold", font_color="#333333",
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
            fontsize=10, fontweight="bold", color="white",
            linespacing=1.15,
        )

    ax.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_file
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Grafo salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grafo de coocorrência de temas")
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
