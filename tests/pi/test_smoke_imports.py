"""Smoke tests: garante que os scripts da pasta pi/ podem ser importados.

Detecta erros de sintaxe e imports quebrados sem rodar o código de verdade.
É a verificação mais barata e captura uma boa fatia dos erros comuns.
"""

import importlib

import pytest

SCRIPTS = [
    "pi.test_webcam",
    "pi.benchmark_webcam",
    "pi.gravar_deteccao",
    "pi.processar_video",
    "pi.stream_deteccao",
]


@pytest.mark.parametrize("modulo", SCRIPTS)
def test_modulo_importa_sem_erro(modulo):
    """Cada script deve poder ser importado sem lançar exceção."""
    importlib.import_module(modulo)


@pytest.mark.parametrize("modulo", SCRIPTS)
def test_modulo_tem_atributos_basicos(modulo):
    """Verifica que constantes esperadas existem nos módulos."""
    mod = importlib.import_module(modulo)
    # Todos os scripts (exceto test_webcam) usam essas constantes
    if modulo != "pi.test_webcam":
        assert hasattr(mod, "MODELO"), f"{modulo} deveria definir MODELO"
        assert hasattr(mod, "IMGSZ"), f"{modulo} deveria definir IMGSZ"
        assert hasattr(mod, "CONF"), f"{modulo} deveria definir CONF"


@pytest.mark.parametrize("modulo", SCRIPTS)
def test_constantes_tem_valores_validos(modulo):
    """IMGSZ deve ser múltiplo de 32 (requisito do YOLO)."""
    mod = importlib.import_module(modulo)
    if hasattr(mod, "IMGSZ"):
        assert mod.IMGSZ % 32 == 0, f"{modulo}.IMGSZ deve ser múltiplo de 32"
    if hasattr(mod, "CONF"):
        assert 0.0 <= mod.CONF <= 1.0, f"{modulo}.CONF deve estar entre 0 e 1"
