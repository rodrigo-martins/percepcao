from pathlib import Path
from typing import Optional, Dict, Any, List
import re
from textwrap import wrap

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



def add_manual_division_lines(extra_divisions, ax, spans, full_xmin, full_xmax, x_label_pos,
                               y_positions=None, line_style='-', color='black', linewidth=1.2,
                               label_pad=0.02, rotate=0, fontsize=9):
    """
    Desenha linhas contínuas (solid) no eixo y para cada divisão definida em extra_divisions
    e posiciona um label à direita do rótulo da subcategoria.

    extra_divisions: lista de tuplas:
        ("Label", start_idx, end_idx)  -> índices inclusivos (0-based)
        ("Label", [i1, i2, ...])      -> lista explícita de índices
    spans: mapping subcategoria -> [idxs] (usado para calcular centro se necessário)
    full_xmin/full_xmax: limites horizontais para a linha (cobre todo o espaço)
    x_label_pos: posição x para desenhar os labels (deve ficar à direita do x_text_base)
    y_positions: array com posições y reais (ex.: np.arange(len(percent))). Se None usa índices diretos.
    label_pad: deslocamento adicional (fração em unidades de eixo x) aplicado à posição do label.
    """
    import numpy as _np
    if y_positions is None:
        # fallback para caso não passe array: usar índices como posições
        y_positions = _np.arange(max((max(v) if isinstance(v, (list,tuple)) else v) for v in spans.values() if v) + 1)

    for item in extra_divisions:
        if len(item) == 3 and isinstance(item[1], int) and isinstance(item[2], int):
            label, start_i, end_i = item
            idxs = list(range(max(0, start_i), min(len(y_positions)-1, end_i) + 1))
        elif len(item) == 2 and isinstance(item[1], (list, tuple)):
            label, idxs = item
            idxs = [k for k in idxs if 0 <= k < len(y_positions)]
        else:
            # formato inesperado: tentar obter pelo nome (span)
            label = item[0]
            idxs = spans.get(label, [])

        if not idxs:
            continue
        # calcular centro vertical da divisão de forma robusta:
        # - se os índices são um intervalo contíguo, usar extremo inicial/final (centro entre eles)
        # - caso contrário, usar média das posições fornecidas
        idxs_sorted = sorted(idxs)
        # colocar a linha *abaixo* do último item do grupo (não no meio):
        # usa a posição do maior índice + 0.5 (fronteira entre esse item e o próximo)
        try:
            y_center = float(y_positions[idxs_sorted[-1]]) + 0.5
        except Exception:
            # fallback robusto: média + 0.5
            y_center = (sum([y_positions[i] for i in idxs_sorted]) / len(idxs_sorted)) + 0.5

        # desenhar linha contínua atravessando todo o gráfico (cobre area das barras e espaço à direita)
        ax.hlines(y_center, full_xmin, full_xmax, colors=color, linewidth=linewidth, linestyles=line_style, zorder=3)

        # label à direita do label de subcategoria: usar ha='left'
        # colocar o label um pouco à direita do x_label_pos (que normalmente já está à direita dos rótulos)
        x_label = x_label_pos + label_pad * (full_xmax - full_xmin)
        ax.text(x_label, y_center, str(label), ha="left", va="center", rotation=rotate,
                fontsize=fontsize, color=color, zorder=4, clip_on=False)


def _orange_shades(base_hex: str, n: int) -> List:
    """Gera n tons do laranja base: menor -> mais claro, maior -> mais escuro."""
    try:
        base_rgb = np.array(mcolors.to_rgb(base_hex))
    except Exception:
        return [base_hex] * max(1, n)
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


def analyze_likert(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Detecta colunas Likert (colunas cujo cabeçalho começa com '['), gera gráfico divergent
    (negativo à esquerda, positivo à direita) e salva em output/likert.png.
    Retorna dict compatível com os outros analisadores.
    """
    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_dir = project_root / "output"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import re
    # localizar colunas Likert (cabecalho iniciando com '[')
    likert_cols = [c for c in df.columns if str(c).strip().startswith("[")]

    # normalizador local (ignora acentuação/pontuação mínima e espaços extras)
    def _norm_local(s: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\dáàâãéèêíïóôõúçÁÀÂÃÉÈÊÍÏÓÔÕÚÇ\-]', ' ', (s or '').lower())).strip()

    def _clean_header(h):
        return re.sub(r"\[|\]", "", str(h)).strip()

    # ordem desejada (exact texts fornecidos)
    desired_order = [
        # Análise Organizacional
        "Entendi como o experiência de aprendizado se conectou com os objetivos da empresa.",
        "Tive o apoio e os recursos necessários (tempo, tecnologia, suporte) da empresa para realizar esta experiência de aprendizado.",
        
        # Análise de Cargos/Tarefas
        # "Soube quem na sua empresa foi o responsável por organizar a experiência de aprendizado.",
        "O conteúdo da experiência de aprendizado atendeu às necessidades da minha função.",
        "O conteúdo da experiência de aprendizado foi adaptado ao meu nível de experiência.",
        "Minha opinião foi considerada na criação da experiência de aprendizado que participei.",
        
        # Características Individuais
        "A carga horária desta experiência de aprendizado prejudicou meu tempo pessoal (descanso, vida social).",
        "A experiência de aprendizado foi importante para me manter competitivo(a) no mercado.",
        "O que aprendi na experiência de aprendizado me motivou a buscar novos conhecimentos.",
        
        # Motivação para o Treinamento
        "A liderança da minha área incentivou ativamente minha participação na experiência de aprendizado.",
        "Tive tempo suficiente durante o expediente para participar da experiência de aprendizado e estudar.",
        "Participei desta experiência de aprendizado mais por obrigação do que por interesse no conteúdo.",
        "A empresa ofereceu incentivos claros para a participação nesta experiência de aprendizado (ex: bônus, folgas, ressarcimento de custos, pontuação em avaliação de desempenho).",
        "A empresa costuma reconhecer os funcionários que se desenvolvem e concluem treinamentos (ex: através de certificados, elogios públicos, comunicados).",

        # Ambiente de Integração e Pré-treinamento
        "A experiência de aprendizado que participei foi bem organizado e estruturado.",
        "Os materiais que recebi para a experiência de aprendizado foram úteis.",

        # Abordagens Específicas de Aprendizagem
        "O ambiente da experiência de aprendizado (físico ou virtual) foi adequado para o aprendizado.",
        "A explicação do instrutor foi clara e fácil de entender.",
        "A atuação do instrutor foi fundamental para o meu aprendizado e motivação.",
        "As atividades durante a experiência de aprendizado foram interessantes e variadas.",

        # Tecnologias de Aprendizagem e Treinamento a Distância
        # Treinamento e Jogos Baseados em Simulação
        
        # Treinamento em Equipe
        "A experiência de aprendizado continha uma parte relevante dedicada ao desenvolvimento de soft skills (habilidades comportamentais).",

        
        # Avaliação do Treinamento
        "Fiquei satisfeito(a) com a qualidade da experiência de aprendizado que participei.",
        "A experiência de aprendizado me ajudou a desenvolver meu raciocínio para resolver problemas.",
        "Aplicar o que aprendi na experiência de aprendizado melhorou meu desempenho.",
        
        # Transferência de Aprendizagem
        "Consigo aplicar no meu trabalho o que aprendi na experiência de aprendizado.",
        "O que aprendi na experiência de aprendizado me deu mais autonomia no trabalho.",
        "O suporte da minha liderança foi um incentivo para eu aplicar o que aprendi.",
        "A experiência de aprendizado que realizei abriu oportunidades de crescimento na empresa (promoção ou aumento salarial)."
    ]
    desired_order = [
        # --- BEFORE TRAINING ---
        # Organizational Analysis
        "Entendi como o experiência de aprendizado se conectou com os objetivos da empresa.",
        "Tive o apoio e os recursos necessários (tempo, tecnologia, suporte) da empresa para realizar esta experiência de aprendizado.",

        # Job/Task Analysis
        # "Soube quem na sua empresa foi o responsável por organizar a experiência de aprendizado.",
        "O conteúdo da experiência de aprendizado atendeu às necessidades da minha função.",
        "O conteúdo da experiência de aprendizado foi adaptado ao meu nível de experiência.",
        "Minha opinião foi considerada na criação da experiência de aprendizado que participei.",

        # --- DURING TRAINING ---
        # Individual Characteristics
        "A carga horária desta experiência de aprendizado prejudicou meu tempo pessoal (descanso, vida social).",
        "A experiência de aprendizado foi importante para me manter competitivo(a) no mercado.",
        "O que aprendi na experiência de aprendizado me motivou a buscar novos conhecimentos.",

        # Training Motivation
        "A liderança da minha área incentivou ativamente minha participação na experiência de aprendizado.",
        "Tive tempo suficiente durante o expediente para participar da experiência de aprendizado e estudar.",
        "Participei desta experiência de aprendizado mais por obrigação do que por interesse no conteúdo.",
        "A empresa ofereceu incentivos claros para a participação nesta experiência de aprendizado (ex: bônus, folgas, ressarcimento de custos, pontuação em avaliação de desempenho).",
        "A empresa costuma reconhecer os funcionários que se desenvolvem e concluem treinamentos (ex: através de certificados, elogios públicos, comunicados).",

        # Training Induction & Pretraining Environment
        "A experiência de aprendizado que participei foi bem organizado e estruturado.",
        "Os materiais que recebi para a experiência de aprendizado foram úteis.",

        # Specific Learning Approaches
        "O ambiente da experiência de aprendizado (físico ou virtual) foi adequado para o aprendizado.",
        "A explicação do instrutor foi clara e fácil de entender.",
        "A atuação do instrutor foi fundamental para o meu aprendizado e motivação.",
        "As atividades durante a experiência de aprendizado foram interessantes e variadas.",

        # Team Training
        "A experiência de aprendizado continha uma parte relevante dedicada ao desenvolvimento de soft skills (habilidades comportamentais).",

        # --- AFTER TRAINING ---
        # Training Evaluation
        "Fiquei satisfeito(a) com a qualidade da experiência de aprendizado que participei.",
        "A experiência de aprendizado me ajudou a desenvolver meu raciocínio para resolver problemas.",
        "Aplicar o que aprendi na experiência de aprendizado melhorou meu desempenho.",

        # Transfer of Training
        "Consigo aplicar no meu trabalho o que aprendi na experiência de aprendizado.",
        "O que aprendi na experiência de aprendizado me deu mais autonomia no trabalho.",
        "O suporte da minha liderança foi um incentivo para eu aplicar o que aprendi.",
        "A experiência de aprendizado que realizei abriu oportunidades de crescimento na empresa (promoção ou aumento salarial)."
    ]

    manual_divisions = [
            ("Organizational Analysis", 0, 1),
            ("Job/Task Analysis", [2,3,4]),
            ("Individual Characteristics", [5,6,7]),
            ("Training Motivation", [9,10,11,12]),
            ("Training Induction and Pretraining Environment", [13,14]),
            ("Specific Learning Approaches", [15,16,17,18]),
            ("Simulation-Based Training and Games", [19]),
            ("Team Training", [20]),
            ("Training Evaluation", [21,22,23]),
            ("Transfer of Training", [24,25,26,27]),
        ]
    manual_divisions = [
            # --- BEFORE TRAINING ---
            ("Organizational Analysis", [0, 1]),
            ("Job/Task Analysis", [2, 3, 4]),
            
            # --- DURING TRAINING ---
            ("Individual Characteristics", [5, 6, 7]),
            ("Training Motivation", [8, 9, 10, 11, 12]),
            ("Training Induction and Pretraining Environment", [13, 14]),
            ("Specific Learning Approaches", [15, 16, 17, 18]),
            ("Team Training", [19]),
            
            # --- AFTER TRAINING ---
            ("Training Evaluation", [20, 21, 22]),
            ("Transfer of Training", [23, 24, 25, 26]),
        ]

    # pergunta a remover explicitamente (se presente)
    remove_q_norm = _norm_local("Soube quem na sua empresa foi o responsável por organizar a experiência de aprendizado.")

    # construir mapa norm -> coluna original
    norm_to_col = { _norm_local(_clean_header(c)): c for c in likert_cols }

    # montar lista ordenada por desired_order (tentativa exata/normais)
    ordered = []
    for q in desired_order:
        nq = _norm_local(q)
        if nq == remove_q_norm:
            continue
        if nq in norm_to_col:
            ordered.append(norm_to_col[nq])
        else:
            # tentativa permissiva: procurar inclusão
            found = None
            for k, col in norm_to_col.items():
                if nq in k or k in nq:
                    found = col
                    break
            if found and found not in ordered:
                ordered.append(found)

    # anexar quaisquer colunas Likert não mapeadas (preservar original order), excluindo a removida
    remaining = [c for c in likert_cols if c not in ordered and _norm_local(_clean_header(c)) != remove_q_norm]
    likert_cols = ordered + remaining
    if not likert_cols:
        return {"column": None, "counts": None, "total": 0, "percentages": {}, "image": None}

    df_likert = df[likert_cols].copy()

    # limpar cabeçalhos e quebrar texto longo para exibição
    cleaned = [re.sub(r"\[|\]", "", str(c)).strip() for c in df_likert.columns]
    cleaned = ["\n".join(wrap(c, width=70)) for c in cleaned]
    df_likert.columns = cleaned

    # ordem esperada das respostas (mantida)
    order = ["Discordo totalmente", "Discordo", "Neutro", "Concordo", "Concordo totalmente"]

    # contagens por pergunta (linhas = pergunta, colunas = respostas)
    counts = df_likert.apply(lambda x: x.value_counts()).fillna(0).reindex(order).T
    # converter para inteiros
    counts = counts.astype(int)

    # porcentagens por linha
    percent = counts.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=1)

    # inverter sinais dos itens negativos para plot divergente
    for neg in ["Discordo totalmente", "Discordo", "Neutro"]:
        if neg in percent.columns:
            percent[neg] = -percent[neg]

    image_path: Optional[Path] = None
    if plt is None:
        return {
            "column": ", ".join(likert_cols),
            "counts": counts,
            "total": int(len(df)),
            "percentages": {},
            "image": None,
        }

    try:
        # cores: usar tons de laranja do projeto e cinza para neutro
        n_shades = 5
        shades = _orange_shades(BASE_ORANGE, n_shades)
        # mapear categorias para cores (extremos mais escuros)
        # shades index: 0..4 (claros->escuros)
        color_map = {
            "Discordo totalmente": shades[1] if len(shades) > 1 else BASE_ORANGE,
            "Discordo": shades[0] if len(shades) > 0 else BASE_ORANGE,
            "Neutro": (0.827, 0.827, 0.827),  # light gray
            "Concordo": shades[-2] if len(shades) > 1 else BASE_ORANGE,
            "Concordo totalmente": shades[-1] if len(shades) > 0 else BASE_ORANGE,
        }

        # plot stacked horizontal divergent bars
        fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(counts))))
        height = 0.6

        # negativo: Discordo totalmente, Discordo, Neutro (neutro fica à esquerda por escolha)
        neg_cols = [c for c in ["Neutro", "Discordo","Discordo totalmente"] if c in percent.columns]
        lefts_neg = np.zeros(len(percent))
        for col in neg_cols:
            vals = percent[col].values
            bars = ax.barh(range(len(percent)), vals, left=lefts_neg, color=color_map.get(col), height=height, label=col, edgecolor="white")
            # anotação com contagem absoluta
            for i, v in enumerate(vals):
                cnt = counts.iloc[i].get(col, 0)
                if cnt:
                    x_pos = lefts_neg[i] + v / 2
                    ax.text(x_pos, i, f"{int(abs(cnt))}", ha="center", va="center", fontsize=8, color="black")
            lefts_neg += vals

        # positivo: Concordo, Concordo totalmente
        pos_cols = [c for c in ["Concordo", "Concordo totalmente"] if c in percent.columns]
        lefts_pos = np.zeros(len(percent))
        for col in pos_cols:
            vals = percent[col].values
            bars = ax.barh(range(len(percent)), vals, left=lefts_pos, color=color_map.get(col), height=height, label=col, edgecolor="white")
            for i, v in enumerate(vals):
                cnt = counts.iloc[i].get(col, 0)
                if cnt:
                    x_pos = lefts_pos[i] + v / 2
                    ax.text(x_pos, i, f"{int(abs(cnt))}", ha="center", va="center", fontsize=8, color="black")
            lefts_pos += vals

        # --- novo: dividir perguntas em subcategorias (apenas subcategorias) e desenhar linhas/labels ---
        # mapa de subcategorias (cada tupla: subcategoria, [perguntas esperadas])
        subcategories = [
            ("Organizational Analysis", [
                "Entendi como o experiência de aprendizado se conectou com os objetivos da empresa.",
                "Tive o apoio e os recursos necessários (tempo, tecnologia, suporte) da empresa para realizar esta experiência de aprendizado."
            ]),
            ("Job/Task Analysis", [
                "Soube quem na sua empresa foi o responsável por organizar a experiência de aprendizado.",
                "O conteúdo da experiência de aprendizado atendeu às necessidades da minha função.",
                "A experiência de aprendizado continha uma parte relevante dedicada ao desenvolvimento de soft skills (habilidades comportamentais).",
                "Minha opinião foi considerada na criação da experiência de aprendizado que participei.",
                "O conteúdo da experiência de aprendizado foi adaptado ao meu nível de experiência.",
                "O que aprendi na experiência de aprendizado me motivou a buscar novos conhecimentos."
            ]),
            ("Individual Characteristics", [
                "A experiência de aprendizado foi importante para me manter competitivo(a) no mercado.",
                "O conteúdo da experiência de aprendizado foi adaptado ao meu nível de experiência.",
                "A carga horária desta experiência de aprendizado prejudicou meu tempo pessoal (descanso, vida social)."
            ]),
            ("Training Motivation", [
                "A liderança da minha área incentivou ativamente minha participação na experiência de aprendizado.",
                "Tive tempo suficiente durante o expediente para participar da experiência de aprendizado e estudar.",
                "Participei desta experiência de aprendizado mais por obrigação do que por interesse no conteúdo.",
                "A empresa ofereceu incentivos claros para a participação nesta experiência de aprendizado (ex: bônus, folgas, ressarcimento de custos, pontuação em avaliação de desempenho).",
                "A empresa costuma reconhecer os funcionários que se desenvolvem e concluem treinamentos (ex: através de certificados, elogios públicos, comunicados)."
            ]),
            ("Training Induction and Pretraining Environment", [
                "A experiência de aprendizado que participei foi bem organizado e estruturado.",
                "Os materiais que recebi para a experiência de aprendizado foram úteis."
            ]),
            ("Specific Learning Approaches", [
                "O ambiente da experiência de aprendizado (físico ou virtual) foi adequado para o aprendizado.",
                "A explicação do instrutor foi clara e fácil de entender.",
                "A atuação do instrutor foi fundamental para o meu aprendizado e motivação.",
                "As atividades durante a experiência de aprendizado foram interessantes e variadas."
            ]),
            ("Team Training", [
                "A experiência de aprendizado continha uma parte relevante dedicada ao desenvolvimento de soft skills (habilidades comportamentais)."
            ]),
            ("Training Evaluation", [
                "Fiquei satisfeito(a) com a qualidade da experiência da experiência de aprendizado que participei.",
                "A experiência de aprendizado me ajudou a desenvolver meu raciocínio para resolver problemas.",
                "Aplicar o que aprendi na experiência de aprendizado melhorou meu desempenho."
            ]),
            ("Transfer of Training", [
                "Consigo aplicar no meu trabalho o que aprendi na experiência de aprendizado.",
                "O que aprendi na experiência de aprendizado me deu mais autonomia no trabalho.",
                "O suporte da minha liderança foi um incentivo para eu aplicar o que aprendi.",
                "A experiência de aprendizado que realizei abriu oportunidades de crescimento na empresa (promoção ou aumento salarial)."
            ]),
        ]

        # normalizador permissivo
        def _norm(s: str) -> str:
            return re.sub(r'\s+', ' ', re.sub(r'[^\w\dáàâãéèêíïóôõúçÁÀÂÃÉÈÊÍÏÓÔÕÚÇ\-]', ' ', s or '').lower()).strip()

        # perguntas originais (sem wrap) e keys exibidas (wrapped)
        original_questions = [re.sub(r'\[|\]', '', str(c)).strip() for c in likert_cols]
        questions_wrapped = percent.index.tolist()

        # mapear cada pergunta original para uma subcategoria (default "Outros")
        sub_for_orig = []
        for q in original_questions:
            nq = _norm(q)
            assigned_sub = "Outros"
            for sub, items in subcategories:
                for it in items:
                    if _norm(it) in nq or _norm(it) == nq:
                        assigned_sub = sub
                        break
                if assigned_sub != "Outros":
                    break
            sub_for_orig.append(assigned_sub)

        # map wrapped -> original e construir lista de subcategorias na ordem exibida
        wrapped_to_orig = {
            "\n".join(wrap(re.sub(r'\[|\]', '', str(c)).strip(), width=70)): re.sub(r'\[|\]', '', str(c)).strip()
            for c in likert_cols
        }
        subs_ordered = []
        # Opção 1 — definições manuais (descomente/edite conforme necessário)
        # Formatos aceitos:
        # ("Label", start_idx, end_idx)  -> índices inclusivos (0-based, na ordem exibida)
        # ("Label", [i1, i2, ...])      -> lista explícita de índices

        if manual_divisions:
            # inicializa tudo como 'Outros'
            subs_ordered = ["Outros"] * len(questions_wrapped)
            for item in manual_divisions:
                if len(item) == 3 and isinstance(item[1], int) and isinstance(item[2], int):
                    label, start_i, end_i = item
                    start_i = max(0, start_i)
                    end_i = min(len(subs_ordered)-1, end_i)
                    for k in range(start_i, end_i + 1):
                        subs_ordered[k] = label
                elif len(item) == 2 and isinstance(item[1], (list, tuple)):
                    label, idxs = item
                    for k in idxs:
                        if 0 <= k < len(subs_ordered):
                            subs_ordered[k] = label
        else:
            # comportamento automático (fallback existente)
            for w in questions_wrapped:
                w_plain = w.replace("\n", " ").strip()
                nw = _norm(w_plain)
                if nw in norm_to_index:
                    idx = norm_to_index[nw]
                    subs_ordered.append(sub_for_orig[idx])
                else:
                    # tentativa por inclusão (mais permissiva)
                    matched = None
                    for norm_q, idx in norm_to_index.items():
                        if norm_q in nw or nw in norm_q:
                            matched = idx
                            break
                    if matched is not None:
                        subs_ordered.append(sub_for_orig[matched])
                    else:
                        subs_ordered.append("Outros")

        # encontrar fronteiras entre subcategorias (linha preta entre subcategorias)
        sub_boundaries = []
        for i in range(len(subs_ordered) - 1):
            if subs_ordered[i] != subs_ordered[i + 1]:
                sub_boundaries.append(i + 0.5)

        # aumentar margem direita para colocar rótulos fora do gráfico
        xmin, xmax = ax.get_xlim()
        x_text_base = xmax + (xmax - xmin) * 0.12
        right_margin = x_text_base + (xmax - xmin) * 0.06
        ax.set_xlim(-100, right_margin)

        # desenhar linhas pretas sólidas nas fronteiras cobrindo todo o espaço horizontal (-100..right_margin)
        full_xmin, full_xmax = ax.get_xlim()

        # definir divisões extras (editar conforme necessário)
        extra_divisions = [
            ("Divisão Especial A", [2, 3, 4]),         # lista de índices (ordem do eixo y)
            ("Divisão Especial B", 8, 10),             # range start..end (inclusivo)
        ]

        # desenhar linhas tracejadas para fronteiras entre subcategorias
        for b in sub_boundaries:
            ax.hlines(b, full_xmin, full_xmax, colors="gray", linewidth=1.0, linestyles="--", zorder=2)

        # agrupar índices por subcategoria para posicionar rótulos verticalmente no lado direito
        from collections import defaultdict
        spans = defaultdict(list)
        for idx, sub in enumerate(subs_ordered):
            spans[sub].append(idx)

        # ordenar subcategorias por primeira aparição para manter ordem visual
        ordered_subs = sorted([s for s in spans.keys() if s != "Outros"], key=lambda s: min(spans[s]))

        # agora que spans e positions existem, preparar y_positions e chamar divisões manuais extras
        import numpy as _np
        y_positions = _np.arange(len(percent))
        add_manual_division_lines(
            extra_divisions, ax, spans, full_xmin, full_xmax,
            x_label_pos=x_text_base + (xmax - xmin) * 0.06,
            y_positions=y_positions, line_style='-', color='black', linewidth=1.2,
            label_pad=0.01, rotate=0, fontsize=9
        )

        # DEBUG: imprimir ordem das subcategorias, posições e perguntas correspondentes
        print("Subcategorias (ordem de aparição) e posições:")
        # questions_wrapped contém os rótulos exibidos na mesma ordem do eixo y
        for rank, sub in enumerate(ordered_subs, start=1):
            idxs = spans[sub]
            print(f"{rank}. {sub} -> posições: {idxs}")
            for i in idxs:
                q_wrapped = questions_wrapped[i] if i < len(questions_wrapped) else ""
                print(f"    posição {i}: {q_wrapped}")
        # também imprimir quaisquer perguntas marcadas como 'Outros'
        outros_idxs = spans.get("Outros", [])
        if outros_idxs:
            print("Outros -> posições:", outros_idxs)
            for i in outros_idxs:
                q_wrapped = questions_wrapped[i] if i < len(questions_wrapped) else ""
                print(f"    posição {i}: {q_wrapped}")

        # desenhar rótulos de subcategoria (rotacionados) fora à direita, alinhados ao centro das perguntas correspondentes
        import numpy as _np
        y_positions = _np.arange(len(percent))
        for sub in ordered_subs:
            idxs = spans[sub]
            if not idxs:
                continue
            y_center = (y_positions[min(idxs)] + y_positions[max(idxs)]) / 2.0
            wrapped_label = "\n".join(wrap(sub, width=20))
            ax.text(x_text_base, y_center, wrapped_label, ha="center", va="center", rotation=0,
                     fontsize=9, color="black", zorder=5)

        # --- fim divisão por subcategorias ---
         # --- fim divisão por categorias ---

        # visual tweaks
        ax.axvline(0, color="black", linewidth=0.8)
        # manter escala percentual na esquerda (-100..100) e reservar espaço à direita para categorias
        right_margin = x_text_base + (xmax - xmin) * 0.06
        ax.set_xlim(-100, right_margin)

        ax.set_yticks(range(len(counts)))
        # colocar as perguntas (labels) à esquerda do gráfico
        ax.set_yticklabels(percent.index, fontsize=9)
        ax.yaxis.tick_left()
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, right=False, labelright=False)
        ax.invert_yaxis()

        ax.set_xlabel("Porcentagem", fontsize=12)
        ax.set_title("Distribuição das respostas — Escala Likert", fontsize=13, weight="bold", pad=18)

        # legenda centralizada acima
        from matplotlib.patches import Patch
        desired = ["Discordo totalmente", "Discordo", "Neutro", "Concordo", "Concordo totalmente"]
        legend_labels = [lab for lab in desired if lab in percent.columns]
        handles = [Patch(facecolor=color_map[lab], edgecolor="white", label=lab) for lab in legend_labels]
        ax.legend(handles, legend_labels, title="", bbox_to_anchor=(0.5, 1.02), loc='upper center', ncol=len(legend_labels), frameon=False)

        plt.tight_layout()
        out_file = out_dir / "likert.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        image_path = out_file
    except Exception:
        image_path = None

    # preparar retorno compacto (main.py espera dict com chaves usuais)
    # construir porcentagens por pergunta (mapping label -> (count, pct)) é complexo para multi-col;
    # mantemos counts e total para inspeção manual.
    return {
        "column": ", ".join(likert_cols),
        "counts": counts,
        "total": int(len(df)),
        "percentages": {},  # vazio por compatibilidade com main.py
        "image": image_path,
    }

if __name__ == "__main__":
    import os
    import sys
    import traceback

    project_root = Path(__file__).resolve().parents[1]
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", str(project_root / "data" / "raw.csv"))
    p = Path(csv_path)
    if not p.exists():
        print("CSV não encontrado:", p)
        raise SystemExit(1)

    try:
        df = pd.read_csv(p, dtype=str, engine="python")
    except Exception as e:
        print("Falha ao ler CSV:", e)
        traceback.print_exc()
        raise SystemExit(1)

    res = analyze_likert(df, out_dir=project_root / "output")
    print({"column": res.get("column"), "total": res.get("total"), "image": res.get("image")})
    if res.get("image"):
        print("Gráfico salvo em:", res["image"])
    else:
        print("Nenhuma imagem gerada. Verifique se matplotlib/numpy estão instalados e se há colunas Likert (cabeçalhos começando com '[').")