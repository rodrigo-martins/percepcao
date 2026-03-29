import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuração ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tratado.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'crosstab_perfil')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Renomear colunas de perfil
col_map = {
    df.columns[1]: 'Tipo',
    df.columns[2]: 'Idade',
    df.columns[3]: 'Gênero',
    df.columns[4]: 'Estado',
    df.columns[5]: 'Instrução',
    df.columns[6]: 'Experiência',
    df.columns[7]: 'Nível',
    df.columns[8]: 'Empresa',
    df.columns[9]: 'Área',
}
df = df.rename(columns=col_map)

# Abreviar valores longos
df['Tipo'] = df['Tipo'].str.split(':').str[0]
df['Instrução'] = df['Instrução'].replace({
    'Pós-graduação completa (Especialização, Mestrado ou Doutorado)': 'Pós completa',
    'Pós-graduação incompleta (Especialização, Mestrado ou Doutorado)': 'Pós incompleta',
    'Ensino Superior completo': 'Superior completo',
    'Ensino Superior incompleto': 'Superior incompleto',
    'Ensino Médio completo': 'Médio completo',
})
df['Experiência'] = df['Experiência'].str.replace('.', '', regex=False)
df['Empresa'] = df['Empresa'].replace({
    '100 ou mais funcionários (Grande empresa)': 'Grande (100+)',
    'De 50 a 99 funcionários (Média empresa)': 'Média (50-99)',
    'De 10 a 49 funcionários (Pequena empresa)': 'Pequena (10-49)',
    'Prefiro não responder': 'N/R',
})

# Faixas de idade
bins = [0, 25, 30, 35, 40, 100]
labels = ['≤25', '26-30', '31-35', '36-40', '41+']
df['Faixa Etária'] = pd.cut(df['Idade'], bins=bins, labels=labels)

# --- Pares de crosstab ---
pares = [
    ('Gênero',       'Nível',        'Gênero x Nível Profissional'),
    ('Gênero',       'Área',         'Gênero x Área de Atuação'),
    ('Gênero',       'Instrução',    'Gênero x Grau de Instrução'),
    ('Gênero',       'Experiência',  'Gênero x Tempo de Experiência'),
    ('Gênero',       'Empresa',      'Gênero x Tamanho da Empresa'),
    ('Nível',        'Experiência',  'Nível Profissional x Experiência'),
    ('Nível',        'Área',         'Nível Profissional x Área de Atuação'),
    ('Tipo',         'Nível',        'Tipo (Obrigatória/Opcional) x Nível'),
    ('Tipo',         'Área',         'Tipo (Obrigatória/Opcional) x Área'),
    ('Faixa Etária', 'Nível',        'Faixa Etária x Nível Profissional'),
    ('Faixa Etária', 'Área',         'Faixa Etária x Área de Atuação'),
    ('Instrução',    'Nível',        'Instrução x Nível Profissional'),
]


def gerar_heatmap(ct, titulo, filename):
    """Gera e salva um heatmap a partir de uma crosstab."""
    # Remover linhas/colunas com poucos dados (ex: 'Prefiro não responder' com 1 resp.)
    ct_plot = ct.copy()
    ct_plot = ct_plot.loc[ct_plot.sum(axis=1) > 1]
    ct_plot = ct_plot.loc[:, ct_plot.sum(axis=0) > 1]

    n_rows, n_cols = ct_plot.shape
    fig_w = max(8, n_cols * 1.6 + 2)
    fig_h = max(4, n_rows * 0.9 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        ct_plot,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Contagem'},
        ax=ax,
    )
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(ct_plot.columns.name, fontsize=11)
    ax.set_ylabel(ct_plot.index.name, fontsize=11)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {path}')


# --- Gerar todos os heatmaps ---
print('Gerando heatmaps de tabulação cruzada...\n')

for col1, col2, titulo in pares:
    ct = pd.crosstab(df[col1], df[col2])
    filename = f'crosstab_{col1.lower().replace(" ", "_")}_{col2.lower().replace(" ", "_")}.png'
    # Limpar caracteres especiais do nome
    for ch in ['ê', 'á', 'ã', 'ç', 'é', 'í']:
        filename = filename.replace(ch, {'ê': 'e', 'á': 'a', 'ã': 'a', 'ç': 'c', 'é': 'e', 'í': 'i'}[ch])
    gerar_heatmap(ct, titulo, filename)

# --- Gráfico-resumo combinado (2x2 dos mais relevantes) ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Cruzamento de Perfil dos Respondentes (n=282)', fontsize=16, fontweight='bold', y=1.01)

combos = [
    ('Gênero',       'Nível',       'Gênero x Nível Profissional'),
    ('Faixa Etária', 'Nível',       'Faixa Etária x Nível Profissional'),
    ('Nível',        'Experiência', 'Nível x Experiência'),
    ('Nível',        'Área',        'Nível x Área de Atuação'),
]

for ax, (c1, c2, titulo) in zip(axes.flat, combos):
    ct = pd.crosstab(df[c1], df[c2])
    ct = ct.loc[ct.sum(axis=1) > 1]
    ct = ct.loc[:, ct.sum(axis=0) > 1]
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5,
                linecolor='white', ax=ax, cbar=False)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
resumo_path = os.path.join(OUTPUT_DIR, 'cruzamento_perfil.png')
fig.savefig(resumo_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  Resumo combinado salvo: {resumo_path}')
print('\nConcluído!')
