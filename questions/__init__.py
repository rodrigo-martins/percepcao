# Tornar `questions` um pacote que exporta automaticamente todas as funções `analyze_*`.
import importlib
import pkgutil
import inspect
from typing import Callable

__all__ = []

def _make_missing(name: str, mod: str) -> Callable:
    def _missing(*args, **kwargs):
        raise ImportError(f"Analisador '{name}' não disponível: falha ao importar o módulo '{mod}'")
    return _missing

# importa todos os módulos do pacote e expõe funções analyze_*
for finder, name, ispkg in pkgutil.iter_modules(__path__):
    try:
        mod = importlib.import_module(f"{__name__}.{name}")
    except Exception:
        # não interrompe a descoberta se um módulo falhar ao importar
        continue
    for attr_name, attr in inspect.getmembers(mod, inspect.isfunction):
        if attr_name.startswith("analyze_"):
            globals()[attr_name] = attr
            __all__.append(attr_name)

# lista de analisadores comumente usados que queremos garantir que exista no pacote
_expected_analyzers = {
    "analyze_genero": "genero",
    "analyze_idade": "idade",
    "analyze_obrig_optional": "obrig_optional",
    "analyze_instrucao": "instrucao",
    "analyze_experiencia": "experiencia",
    "plot_respondentes_por_estado": "mapa_estados",  # função de mapa (nome não começa com analyze_)
}

for func_name, module_name in _expected_analyzers.items():
    if func_name in globals():
        # já disponível via descoberta automática
        if func_name not in __all__:
            __all__.append(func_name)
        continue
    # tentar importar explicitamente o módulo e obter o símbolo
    try:
        mod = importlib.import_module(f"{__name__}.{module_name}")
        if hasattr(mod, func_name):
            globals()[func_name] = getattr(mod, func_name)
            __all__.append(func_name)
            continue
        # algumas funções podem ter nomes diferentes (ex.: plot_respondentes_por_estado)
        # já estamos tentando pelo nome previsto; se não existir, criar placeholder
    except Exception:
        # falhou ao importar o módulo; criar placeholder para evitar ImportError ao usar "from questions import ...".
        pass
    # criar placeholder callable que levantará erro informativo quando chamado
    globals()[func_name] = _make_missing(func_name, module_name)
    __all__.append(func_name)

# fallback explícito para manter compatibilidade e garantir availability
try:
    from .genero import analyze_genero  # type: ignore
    globals()["analyze_genero"] = analyze_genero
    if "analyze_genero" not in __all__:
        __all__.append("analyze_genero")
except Exception:
    # se der erro, será silencioso — a descoberta anterior já tentou importar o módulo
    pass