import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tratado.csv')

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
qq2_col = df.columns[38]
df = df.rename(columns=col_map)

# Abreviar valores
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

# --- Filtrar apenas respondentes válidos de QQ2 ---
# Excluir nulos
df_qq2 = df[df[qq2_col].notna()].copy()

# Excluir respostas inválidas
excluir = ['-', 'Neutro', 'Nenhuma', 'Não sei', 'não tenho',
           'Sem comentários!', 'No momento, nao tenho nada a acrescentar']
excluir_lower = [e.lower().strip() for e in excluir]
df_qq2 = df_qq2[~df_qq2[qq2_col].str.strip().str.lower().isin(excluir_lower)]

n = len(df_qq2)
print(f'=== Perfil Sociodemográfico dos Respondentes da QQ2 (n={n}) ===\n')


def freq(col, label=None):
    s = df_qq2[col].value_counts()
    pct = (s / n * 100).round(1)
    print(f'--- {label or col} ---')
    for val, cnt in s.items():
        print(f'  {val}: {cnt} ({pct[val]}%)')
    print()


# Gênero
freq('Gênero')

# Idade
print(f'--- Idade ---')
print(f'  Média: {df_qq2["Idade"].mean():.1f} anos')
print(f'  Mediana: {df_qq2["Idade"].median():.1f} anos')
print(f'  DP: {df_qq2["Idade"].std():.1f}')
print(f'  Min: {df_qq2["Idade"].min()}, Max: {df_qq2["Idade"].max()}')
print()

# Estado
freq('Estado')

# Instrução
freq('Instrução')

# Experiência
freq('Experiência')

# Nível profissional
freq('Nível', 'Nível Profissional')

# Empresa
freq('Empresa', 'Tamanho da Empresa')

# Área
freq('Área', 'Área de Atuação')

# Tipo (obrigatória/opcional)
freq('Tipo', 'Tipo de Participação')

# Instrução agrupada: superior + pós
sup = df_qq2['Instrução'].isin(['Superior completo', 'Superior incompleto']).sum()
pos = df_qq2['Instrução'].isin(['Pós completa', 'Pós incompleta']).sum()
print(f'--- Instrução (agrupada) ---')
print(f'  Ensino superior (completo+incompleto): {sup} ({sup/n*100:.1f}%)')
print(f'  Pós-graduação (completa+incompleta): {pos} ({pos/n*100:.1f}%)')
print()

# Senioridade agrupada
senior_labels = ['Sênior', 'Especialista', 'Gerente']
senior_count = df_qq2['Nível'].isin(senior_labels).sum()
print(f'--- Senioridade (Sênior + Especialista + Gerente/Liderança) ---')
print(f'  {senior_count} ({senior_count/n*100:.1f}%)')
