from pathlib import Path
from typing import Optional
import sys
import re
import unicodedata

def _detect_likert_columns(df):
    return [c for c in df.columns if str(c).strip().startswith("[")]

def _build_mapping():
    # mapeamentos estritos conforme informado (usar lowercase para casar com lookup)
    return {
        "discordo totalmente": 1,
        "discordo": 2,
        "neutro": 3,
        "concordo": 4,
        "concordo totalmente": 5,
    }

def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    # normalize unicode (remove accents)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    s = s.lower()
    # remove extra punctuation except digits and spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _map_series(s, mapping):
    import pandas as pd
    def map_val(v):
        if pd.isna(v):
            return -1
        txt = str(v).strip()
        if txt == "":
            return -1
        # try leading numeric (e.g. "1", "1 - Discordo", "1.0")
        m = re.match(r"^\s*([1-5])\b", txt)
        if m:
            n = int(m.group(1))
            return max(0, n-1)
        # try any digit in string
        m2 = re.search(r"([1-5])", txt)
        if m2 and len(txt) <= 3:  # likely just number or short code
            n = int(m2.group(1))
            return max(0, n-1)
        # normalize and match
        norm = _normalize_text(txt)
        if norm in mapping:
            return mapping[norm]
        # substring match (e.g. "concordo parcialmente" -> contains "concordo")
        for k, v in mapping.items():
            if k in norm:
                return v
        return -1
    return s.map(map_val).astype(int)

def process(csv_path: Optional[str] = None, out_dat: Optional[str] = None,
            drop_question_label: str = "Qual é a sua principal área de atuação na engenharia de software no momento?",
            drop_value: str = "Recursos Humanos"):
    import pandas as pd
    import numpy as np

    csv_path = Path(csv_path or "data/raw.csv")
    out_dat = Path(out_dat or "output/likert.dat")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, engine="python")
    likert_cols = _detect_likert_columns(df)

    # DEBUG: mostrar colunas detectadas inicialmente
    print(f"[DEBUG] colunas totais no CSV: {len(df.columns)}")
    print("[DEBUG] primeiras colunas (head 20):")
    for i, c in enumerate(list(df.columns)[:20], start=1):
        print(f"  {i:02d}: {c}")
    print(f"[DEBUG] colunas Likert detectadas: {len(likert_cols)}")
    for i, c in enumerate(likert_cols, start=1):
        print(f"  L{i:02d}: {c}")

    # remover coluna específica, se existir
    remove_label = "Soube quem na sua empresa foi o responsável por organizar a experiência de aprendizado."
    norm_remove = _normalize_text(remove_label)
    to_remove = []
    for c in likert_cols:
        # remover colchetes e normalizar para comparação
        c_plain = re.sub(r'^\[|\]$', '', str(c)).strip()
        if _normalize_text(c_plain) == norm_remove:
            to_remove.append(c)
    if to_remove:
        df = df.drop(columns=to_remove, errors="ignore")
        likert_cols = [c for c in likert_cols if c not in to_remove]
        print(f"[DEBUG] Removida(s) coluna(s) Likert: {to_remove}")

    # DEBUG: após remoção
    print(f"[DEBUG] colunas Likert após remoção: {len(likert_cols)}")
    for i, c in enumerate(likert_cols, start=1):
        print(f"  A{i:02d}: {c}")

    # ordenar colunas Likert conforme desired_order (mantém quaisquer colunas não listadas ao final)
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
    # construir mapa normalizado das colunas detectadas -> coluna original
    norm_to_col = {}
    for c in likert_cols:
        c_plain = re.sub(r'^\[|\]$', '', str(c)).strip()
        norm = _normalize_text(c_plain)
        norm_to_col[norm] = c

    ordered = []
    used = set()
    for label in desired_order:
        nl = _normalize_text(label)
        if nl in norm_to_col:
            col = norm_to_col[nl]
            if col not in used:
                ordered.append(col)
                used.add(col)
                print(f"  ✓ Matched: '{label[:50]}...' -> {col}")
            else:
                print(f"  ⚠ Duplicado (já usado): '{label[:50]}...'")
        else:
            print(f"  ✗ NÃO ENCONTRADO: '{label[:50]}...'")
            print(f"     Procurou norm: '{nl}'")
            # debug: mostrar os 3 mais similares
            from difflib import SequenceMatcher
            similar = sorted(norm_to_col.keys(), key=lambda k: SequenceMatcher(None, nl, k).ratio(), reverse=True)[:3]
            print(f"     Similares: {similar}")

    # acrescentar quaisquer colunas restantes não listadas em desired_order, preservando ordem original
    for c in likert_cols:
        if c not in used:
            print(f"  + Adicionada ao final (não em desired_order): {c}")
            ordered.append(c)
    likert_cols = ordered
    print(f"Ordem Likert aplicada: {len(likert_cols)} colunas (as não listadas em desired_order foram adicionadas ao final).")

    # DEBUG: mostrar mapeamento normalizado -> original antes da ordenação
    print("[DEBUG] mapa normalizado -> coluna original (ex):")
    for k, v in list(norm_to_col.items())[:40]:
        print(f"  {k} -> {v}")
    print(f"[DEBUG] total colunas após ordenação: {len(likert_cols)}")
    for i, c in enumerate(likert_cols, start=1):
        print(f"  O{i:02d}: {c}")

    if not likert_cols:
        print("Nenhuma coluna Likert detectada (cabeçalhos começando com '['). Saindo.")
        return

    # localizar coluna de filtro (caso o header seja similar)
    found_col = None
    key_norm = _normalize_text(drop_question_label)
    for c in df.columns:
        if _normalize_text(str(c)) == key_norm:
            found_col = c
            break
    if found_col is None:
        for c in df.columns:
            if key_norm in _normalize_text(str(c)):
                found_col = c
                break

    if found_col is not None:
        before = len(df)
        df = df[df[found_col].fillna("").astype(str).str.strip().str.lower() != drop_value.strip().lower()]
        after = len(df)
        print(f"Removidas {before-after} linhas onde '{found_col}' == '{drop_value}'.")
    else:
        print(f"A coluna para filtro ('{drop_question_label}') não foi encontrada; nenhum dado removido.")

    mapping = _build_mapping()
    converted_cols = []
    for c in likert_cols:
        series = df[c].fillna("")
        mapped = _map_series(series, mapping)
        converted_cols.append(mapped)
        print(f"Coluna '{c}': valores únicos após mapeamento -> {sorted(list(pd.Series(mapped).unique()))[:10]}")

    if not converted_cols:
        print("Nenhuma coluna convertida. Saindo.")
        return

    arr = np.vstack([s.values for s in converted_cols]).T  # shape (n_rows, n_cols)
    out_dat.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dat, arr, fmt="%d", delimiter=" ")
    print(f"Salvo {arr.shape[0]} linhas x {arr.shape[1]} colunas em: {out_dat}")
    return out_dat

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        process(csv, out)
    except Exception as e:
        print("Erro:", e)
        raise