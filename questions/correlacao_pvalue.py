from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.cm as cm
from matplotlib.colors import Normalize

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

# ==============================================================================
# CONFIGURAÇÃO OBRIGATÓRIA
# ==============================================================================
# Insira aqui o número total de respondentes da sua pesquisa
TAMANHO_AMOSTRA = 282  
# ==============================================================================

def get_text_color(value: float, cmap_name: str = "RdBu_r") -> str:
    """
    Determina se o texto deve ser branco ou preto baseado no valor (cor de fundo).
    Usa uma abordagem simples: valores com |r| > threshold terão fundo escuro
    """
    # Threshold: correlações acima desse valor absoluto têm fundo escuro
    return 'white' if abs(value) > 0.45 else 'black'

def calcular_matriz_p_values(corr_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Calcula a aproximação dos p-values para a hipótese nula (rho=0).
    Nota: Para correlações policóricas, este teste é uma aproximação que assume
    distribuição assintótica normal do estimador.
    """
    df = n - 2
    r = corr_matrix.to_numpy()
    
    # Clip para evitar erros numéricos nos extremos
    r = np.clip(r, -0.999999, 0.999999)
    
    # Estatística t
    t_stat = r * np.sqrt(df / (1 - r**2))
    
    # P-value bilateral (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # Zera a diagonal principal
    np.fill_diagonal(p_values, 0.0)
    
    return pd.DataFrame(p_values, index=corr_matrix.index, columns=corr_matrix.columns)

def criar_heatmap_correlacao(corr_matrix: pd.DataFrame, n_samples: int, out_path: Optional[Path] = None, 
                               cmap: str = "RdBu_r", figsize: tuple = (16, 14), 
                               annot_fontsize: int = 8) -> Optional[Path]:
    
    if plt is None or sns is None:
        print("Bibliotecas gráficas (matplotlib/seaborn) não disponíveis.")
        return None
    
    out_path = Path(out_path) if out_path is not None else Path("output/correlacao_policorica_pvalue.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Calcular P-Values
    p_matrix = calcular_matriz_p_values(corr_matrix, n_samples)
    
    # 2. Preparar Rótulos (Q1, Q2...)
    n = len(corr_matrix)
    new_labels = [f"Q{i+1}" for i in range(n)]
    
    corr_display = corr_matrix.copy()
    corr_display.index = new_labels
    corr_display.columns = new_labels
    
    # 3. Criar Matriz de Texto para o Heatmap (apenas com R)
    annot_labels = pd.DataFrame(index=new_labels, columns=new_labels)
    
    for r_idx in new_labels:
        for c_idx in new_labels:
            r_val = corr_display.loc[r_idx, c_idx]
            # Apenas o valor de R na anotação principal
            annot_labels.loc[r_idx, c_idx] = f"{r_val:.3f}"

    # 4. Configurar Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_display, 
                annot=annot_labels, 
                fmt="",             
                cmap=cmap, 
                center=0.0,
                vmin=-1, vmax=1, 
                ax=ax, 
                cbar=False,
                annot_kws={"fontsize": annot_fontsize + 4},  # R com fonte maior
                linewidths=0.5, linecolor="white")
    
    # Adicionar p-values customizados com fonte menor
    pval_fontsize = annot_fontsize
    for i, r_idx in enumerate(new_labels):
        for j, c_idx in enumerate(new_labels):
            if r_idx != c_idx:  # Não adicionar na diagonal
                p_val = p_matrix.iloc[i, j]
                if p_val < 0.001:
                    p_text = "(<.001)"
                else:
                    p_text = f"({p_val:.3f})"
                
                # Determinar cor do texto baseado no valor de correlação
                r_val = corr_display.iloc[i, j]
                text_color = get_text_color(r_val, cmap_name=cmap)
                
                # Adicionar o texto do p-value abaixo do R
                ax.text(j + 0.5, i + .85, p_text, ha='center', va='center', 
                       fontsize=pval_fontsize, color=text_color, weight='normal')
    
    # Ajustes estéticos
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    # --- CORREÇÃO DA NOTA DE RODAPÉ ---
    # nota_rodape = (
    #     f"Nota: Valores superiores indicam o Coeficiente de Correlação Policórica (ρ). "
    #     f"Valores entre parênteses indicam a significância estatística (p-value) para N={n_samples}."
    # )
    nota_rodape = ''
    plt.figtext(0.5, 0.02, nota_rodape, wrap=True, horizontalalignment='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Heatmap salvo em: {out_path}")
    return out_path


def gerar_matriz_exemplo() -> pd.DataFrame:
    # Dados da matriz fornecida
    data = [
        [1.000, 0.509, 0.607, 0.401, 0.338, -0.155, 0.486, 0.440, 0.250, 0.367, -0.258, 0.150, 0.295, 0.492, 0.529, 0.474, 0.424, 0.452, 0.518, 0.331, 0.562, 0.500, 0.410, 0.476, 0.439, 0.272, 0.165],
        [0.509, 1.000, 0.483, 0.421, 0.303, -0.205, 0.302, 0.302, 0.410, 0.562, -0.203, 0.307, 0.432, 0.581, 0.576, 0.509, 0.417, 0.365, 0.394, 0.353, 0.522, 0.362, 0.338, 0.370, 0.403, 0.408, 0.179],
        [0.607, 0.483, 1.000, 0.545, 0.552, -0.108, 0.576, 0.568, 0.304, 0.351, -0.422, 0.225, 0.358, 0.468, 0.555, 0.446, 0.457, 0.556, 0.609, 0.547, 0.654, 0.631, 0.606, 0.595, 0.614, 0.411, 0.365],
        [0.401, 0.421, 0.545, 1.000, 0.620, -0.138, 0.475, 0.512, 0.264, 0.278, -0.286, 0.228, 0.370, 0.430, 0.508, 0.367, 0.355, 0.470, 0.514, 0.476, 0.471, 0.501, 0.450, 0.332, 0.453, 0.429, 0.313],
        [0.338, 0.303, 0.552, 0.620, 1.000, -0.091, 0.332, 0.412, 0.253, 0.264, -0.282, 0.213, 0.383, 0.338, 0.401, 0.359, 0.349, 0.427, 0.459, 0.557, 0.493, 0.472, 0.377, 0.325, 0.429, 0.397, 0.360],
        [-0.155, -0.205, -0.108, -0.138, -0.091, 1.000, 0.055, -0.059, -0.109, -0.308, 0.308, 0.098, -0.067, -0.158, -0.106, -0.003, -0.146, -0.040, 0.011, -0.126, -0.094, 0.023, -0.045, -0.031, 0.005, -0.100, 0.210],
        [0.486, 0.302, 0.576, 0.475, 0.332, 0.055, 1.000, 0.703, 0.375, 0.251, -0.508, 0.332, 0.423, 0.348, 0.540, 0.357, 0.317, 0.466, 0.538, 0.295, 0.608, 0.684, 0.648, 0.541, 0.577, 0.439, 0.488],
        [0.440, 0.302, 0.568, 0.512, 0.412, -0.059, 0.703, 1.000, 0.275, 0.315, -0.546, 0.268, 0.484, 0.461, 0.627, 0.464, 0.444, 0.595, 0.660, 0.345, 0.721, 0.827, 0.779, 0.643, 0.702, 0.520, 0.554],
        [0.250, 0.410, 0.304, 0.264, 0.253, -0.109, 0.375, 0.275, 1.000, 0.291, -0.151, 0.331, 0.472, 0.333, 0.343, 0.335, 0.393, 0.302, 0.330, 0.110, 0.334, 0.334, 0.332, 0.363, 0.368, 0.466, 0.223],
        [0.367, 0.562, 0.351, 0.278, 0.264, -0.308, 0.251, 0.315, 0.291, 1.000, -0.177, 0.388, 0.478, 0.420, 0.444, 0.371, 0.361, 0.378, 0.372, 0.211, 0.440, 0.376, 0.335, 0.419, 0.353, 0.411, 0.284],
        [-0.258, -0.203, -0.422, -0.286, -0.282, 0.308, -0.508, -0.546, -0.151, -0.177, 1.000, -0.167, -0.218, -0.267, -0.340, -0.208, -0.256, -0.343, -0.409, -0.201, -0.523, -0.480, -0.446, -0.386, -0.387, -0.335, -0.249],
        [0.150, 0.307, 0.225, 0.228, 0.213, 0.098, 0.332, 0.268, 0.331, 0.388, -0.167, 1.000, 0.502, 0.234, 0.287, 0.279, 0.225, 0.279, 0.286, 0.223, 0.302, 0.398, 0.313, 0.237, 0.294, 0.382, 0.403],
        [0.295, 0.432, 0.358, 0.370, 0.383, -0.067, 0.423, 0.484, 0.472, 0.478, -0.218, 0.502, 1.000, 0.474, 0.564, 0.480, 0.416, 0.510, 0.534, 0.323, 0.538, 0.470, 0.497, 0.448, 0.507, 0.586, 0.536],
        [0.492, 0.581, 0.468, 0.430, 0.338, -0.158, 0.348, 0.461, 0.333, 0.420, -0.267, 0.234, 0.474, 1.000, 0.763, 0.735, 0.745, 0.629, 0.611, 0.361, 0.728, 0.558, 0.463, 0.475, 0.479, 0.389, 0.271],
        [0.529, 0.576, 0.555, 0.508, 0.401, -0.106, 0.540, 0.627, 0.343, 0.444, -0.340, 0.287, 0.564, 0.763, 1.000, 0.725, 0.650, 0.707, 0.730, 0.335, 0.774, 0.632, 0.570, 0.577, 0.576, 0.464, 0.428],
        [0.474, 0.509, 0.446, 0.367, 0.359, -0.003, 0.357, 0.464, 0.335, 0.371, -0.208, 0.279, 0.480, 0.735, 0.725, 1.000, 0.751, 0.681, 0.679, 0.318, 0.724, 0.569, 0.429, 0.471, 0.497, 0.350, 0.333],
        [0.424, 0.417, 0.457, 0.355, 0.349, -0.146, 0.317, 0.444, 0.393, 0.361, -0.256, 0.225, 0.416, 0.745, 0.650, 0.751, 1.000, 0.746, 0.665, 0.312, 0.722, 0.559, 0.420, 0.468, 0.435, 0.377, 0.241],
        [0.452, 0.365, 0.556, 0.470, 0.427, -0.040, 0.466, 0.595, 0.302, 0.378, -0.343, 0.279, 0.510, 0.629, 0.707, 0.681, 0.746, 1.000, 0.837, 0.426, 0.785, 0.714, 0.600, 0.544, 0.611, 0.494, 0.464],
        [0.518, 0.394, 0.609, 0.514, 0.459, 0.011, 0.538, 0.660, 0.330, 0.372, -0.409, 0.286, 0.534, 0.611, 0.730, 0.679, 0.665, 0.837, 1.000, 0.442, 0.804, 0.735, 0.673, 0.578, 0.672, 0.533, 0.493],
        [0.331, 0.353, 0.547, 0.476, 0.557, -0.126, 0.295, 0.345, 0.110, 0.211, -0.201, 0.223, 0.323, 0.361, 0.335, 0.318, 0.312, 0.426, 0.442, 1.000, 0.397, 0.456, 0.377, 0.345, 0.414, 0.355, 0.285],
        [0.562, 0.522, 0.654, 0.471, 0.493, -0.094, 0.608, 0.721, 0.334, 0.440, -0.523, 0.302, 0.538, 0.728, 0.774, 0.724, 0.722, 0.785, 0.804, 0.397, 1.000, 0.806, 0.675, 0.632, 0.653, 0.548, 0.470],
        [0.500, 0.362, 0.631, 0.501, 0.472, 0.023, 0.684, 0.827, 0.334, 0.376, -0.480, 0.398, 0.470, 0.558, 0.632, 0.569, 0.559, 0.714, 0.735, 0.456, 0.806, 1.000, 0.758, 0.645, 0.739, 0.577, 0.581],
        [0.410, 0.338, 0.606, 0.450, 0.377, -0.045, 0.648, 0.779, 0.332, 0.335, -0.446, 0.313, 0.497, 0.463, 0.570, 0.429, 0.420, 0.600, 0.673, 0.377, 0.675, 0.758, 1.000, 0.828, 0.884, 0.633, 0.607],
        [0.476, 0.370, 0.595, 0.332, 0.325, -0.031, 0.541, 0.643, 0.363, 0.419, -0.386, 0.237, 0.448, 0.475, 0.577, 0.471, 0.468, 0.544, 0.578, 0.345, 0.632, 0.645, 0.828, 1.000, 0.800, 0.533, 0.483],
        [0.439, 0.403, 0.614, 0.453, 0.429, 0.005, 0.577, 0.702, 0.368, 0.353, -0.387, 0.294, 0.507, 0.479, 0.576, 0.497, 0.435, 0.611, 0.672, 0.414, 0.653, 0.739, 0.884, 0.800, 1.000, 0.684, 0.631],
        [0.272, 0.408, 0.411, 0.429, 0.397, -0.100, 0.439, 0.520, 0.466, 0.411, -0.335, 0.382, 0.586, 0.389, 0.464, 0.350, 0.377, 0.494, 0.533, 0.355, 0.548, 0.577, 0.633, 0.533, 0.684, 1.000, 0.625],
        [0.165, 0.179, 0.365, 0.313, 0.360, 0.210, 0.488, 0.554, 0.223, 0.284, -0.249, 0.403, 0.536, 0.271, 0.428, 0.333, 0.241, 0.464, 0.493, 0.285, 0.470, 0.581, 0.607, 0.483, 0.631, 0.625, 1.000],
    ]
    
    df = pd.DataFrame(data, index=[f"V{i+1}" for i in range(27)],
                      columns=[f"V{i+1}" for i in range(27)])
    return df

if __name__ == "__main__":
    corr = gerar_matriz_exemplo()
    criar_heatmap_correlacao(corr, n_samples=TAMANHO_AMOSTRA, out_path=Path("output/correlacao_heatmap_policorica_pvalue.png"))