# filepath: questions/afe.py
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# dependências de análise fatorial / plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    try:
        import seaborn as sns
    except Exception:
        sns = None
except Exception as _e:
    matplotlib = None
    plt = None
    np = None
    FactorAnalyzer = None
    calculate_kmo = None
    calculate_bartlett_sphericity = None
    sns = None

BASE_ORANGE = "#ff6002"


def load_likert_data(path: Path) -> pd.DataFrame:
    """
    Lê arquivo whitespace-delimited (output/likert.dat) e retorna DataFrame.
    Assumimos linhas = respondentes, colunas = itens Likert.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    df = pd.read_csv(p, sep=r"\s+", header=None, engine="python")
    # nomear colunas Q1..QN
    df.columns = [f"Q{i+1}" for i in range(df.shape[1])]
    # forçar numérico e remover linhas incompletas
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    return df


def analisar_kmo_bartlett(df: pd.DataFrame) -> dict:
    """Calcula KMO e Bartlett (retorna dict)."""
    if calculate_kmo is None or calculate_bartlett_sphericity is None:
        raise ImportError("factor_analyzer não disponível")
    kmo_all, kmo_model = calculate_kmo(df)
    bartlett_chi, bartlett_p = calculate_bartlett_sphericity(df)
    return {"KMO_model": float(kmo_model), "KMO_per_item": dict(zip(df.columns.tolist(), list(map(float, kmo_all)))), "Bartlett_chi2": float(bartlett_chi), "Bartlett_pvalue": float(bartlett_p)}


def definir_numero_fatores(df: pd.DataFrame, out_dir: Path, plot: bool = True) -> pd.Series:
    """Retorna autovalores (e salva scree plot em out_dir/afe_scree.png quando plot=True)."""
    if FactorAnalyzer is None:
        raise ImportError("factor_analyzer não disponível")
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df)
    eigenvals, _ = fa.get_eigenvalues()
    ev = pd.Series(eigenvals, index=[f"Fator{i+1}" for i in range(len(eigenvals))])
    if plot and plt is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(eigenvals) + 1), eigenvals, marker="o", lw=1.5, color=BASE_ORANGE)
        ax.axhline(1.0, color="red", linestyle="--", lw=1)
        ax.set_xlabel("Fator")
        ax.set_ylabel("Autovalor")
        ax.set_title("Scree Plot (Autovalores)")
        ax.grid(alpha=0.3)
        fig_path = out_dir / "afe_scree.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ev


def rodar_efa(df: pd.DataFrame, n_fatores: int, rotacao: str = "varimax", min_loading_threshold: float = 0.5) -> Dict[str, Any]:
    """Executa EFA e retorna cargas, comunalidades e variância explicada.
    Remove (filtra) variáveis cuja maior carga absoluta em todos os fatores for menor que min_loading_threshold.
    """
    if FactorAnalyzer is None:
        raise ImportError("factor_analyzer não disponível")
    fa = FactorAnalyzer(n_factors=n_fatores, rotation=rotacao)
    fa.fit(df)
    loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=[f"Fator {i+1}" for i in range(n_fatores)])
    communalities = pd.Series(fa.get_communalities(), index=df.columns)
    fv = fa.get_factor_variance()
    variance = pd.DataFrame({"Variance": fv[0], "% Variance": fv[1], "% Cumulative": fv[2]}, index=[f"Fator {i+1}" for i in range(n_fatores)])

    # Filtrar cargas pequenas: manter somente |carga| >= min_loading_threshold
    filtered_loadings = loadings.copy()
    filtered_loadings[filtered_loadings.abs() < min_loading_threshold] = 0.0

    # remover variáveis cuja máxima carga absoluta é 0 (i.e. todas < threshold)
    removed_variables = [idx for idx in filtered_loadings.index if filtered_loadings.loc[idx].abs().max() == 0.0]
    if removed_variables:
        filtered_loadings = filtered_loadings.drop(index=removed_variables)
        communalities = communalities.drop(index=removed_variables, errors="ignore")

    return {
        "loadings": loadings,
        "filtered_loadings": filtered_loadings,
        "removed_variables": removed_variables,
        "comunalidades": communalities,
        "variancia": variance,
        "fa_object": fa
    }


def interpretar_cargas(loadings: pd.DataFrame, threshold: float = 0.40) -> Dict[str, list]:
    """Retorna variáveis com carga >= threshold por fator."""
    interpretacao = {}
    for fator in loadings.columns:
        variaveis = loadings.index[loadings[fator].abs() >= threshold].tolist()
        interpretacao[fator] = variaveis
    return interpretacao


def _save_loadings_heatmap(loadings: pd.DataFrame, out_dir: Path, cmap: str = "Oranges") -> Optional[Path]:
    """Salva heatmap das cargas em out_dir/afe_loadings.png."""
    if plt is None or np is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(loadings)), max(4, 0.6 * loadings.shape[1])))
    if sns is not None:
        sns.heatmap(loadings, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax, cbar_kws={"shrink": 0.6})
    else:
        im = ax.imshow(loadings.values, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
        ax.set_yticks(range(len(loadings)))
        ax.set_yticklabels(loadings.index)
        ax.set_xticks(range(len(loadings.columns)))
        ax.set_xticklabels(loadings.columns, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, shrink=0.6)
        for (i, j), val in np.ndenumerate(loadings.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Loadings (Cargas fatoriais)")
    plt.tight_layout()
    out_file = out_dir / "afe_loadings.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file


def analise_fatorial_exploratoria_from_file(path: Path, n_fatores: Optional[int] = None, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Pipeline completo:
    - carrega arquivo likert (whitespace)
    - KMO e Bartlett
    - determina autovalores (scree) e salva scree plot
    - se n_fatores None, usa regra Kaiser (eigen>1)
    - executa EFA, salva heatmap de cargas e retorna resultado
    """
    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(out_dir) if out_dir is not None else project_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_likert_data(Path(path))
    result: Dict[str, Any] = {"n_rows": int(df.shape[0]), "n_items": int(df.shape[1])}

    # KMO / Bartlett
    try:
        result["adequacao"] = analisar_kmo_bartlett(df)
    except Exception as e:
        result["adequacao_error"] = str(e)

    # eigenvalues + scree
    try:
        ev = definir_numero_fatores(df, out_dir=out_dir, plot=True)
        result["eigenvalues"] = ev
        if n_fatores is None:
            n_fatores = int((ev > 1.0).sum())
            if n_fatores < 1:
                n_fatores = 1
        result["suggested_n_factors_kaiser"] = int(n_fatores)
    except Exception as e:
        result["eigen_error"] = str(e)
        n_fatores = n_fatores or 1

    # rodar efa
    try:
        # chamar com rotação oblíqua promax:
        efa_res = rodar_efa(df, n_fatores=n_fatores, rotacao="varimax")
        result.update({"cargas": efa_res["loadings"], "comunalidades": efa_res["comunalidades"], "variancia": efa_res["variancia"]})
        # salvar heatmap
        try:
            img = _save_loadings_heatmap(efa_res["loadings"], out_dir=out_dir)
            result["image_loadings"] = img
        except Exception as e:
            result["image_loadings_error"] = str(e)
        # interpretar
        try:
            result["interpretacao"] = interpretar_cargas(efa_res["loadings"])
        except Exception as e:
            result["interpretacao_error"] = str(e)
    except Exception as e:
        result["efa_error"] = str(e)

    return result


# execução direta: python -m questions.afe ou python questions/afe.py <path> [n_fatores]
if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/likert.dat")
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path("output")
    try:
        res = analise_fatorial_exploratoria_from_file(p, n_fatores=nf, out_dir=out)
        keys = {k: ("<DataFrame>" if k in ("cargas","comunalidades","variancia") else v) for k, v in res.items()}
        from pprint import pformat
        summary = {k: (type(v).__name__ if hasattr(v, 'shape') else v) for k, v in res.items()}
        print("AFE resultado resumido:")
        print(pformat(summary, indent=2))
        if res.get("image_loadings"):
            print("Heatmap salvo em:", res["image_loadings"])
        if res.get("eigenvalues") is not None:
            print("Scree salvo em:", out / "afe_scree.png")
    except Exception as e:
        print("Erro:", e)