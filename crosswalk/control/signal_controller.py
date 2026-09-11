"""Controlador de sinal: a interface entre a decisão e o hardware do semáforo.

Define uma interface abstrata (SignalController) para que o mesmo pipeline
rode tanto no PC (com um mock que só registra/imprime o que faria) quanto no
Raspberry Pi de produção (com uma implementação real via GPIO/relé).

Essa separação é o que permite desenvolver e testar toda a lógica de decisão
no PC, sem hardware nenhum, e só trocar esta peça na hora do deploy final.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class SignalController(ABC):
    """Interface que qualquer controlador de semáforo (real ou mock) deve implementar."""

    @abstractmethod
    def estender_verde(self, segundos: float) -> None:
        """Solicita a extensão do tempo de verde em `segundos`.

        Implementações reais devem ser idempotentes/seguras a chamadas repetidas
        (o pipeline chama isso a cada frame enquanto há cadeirante na faixa).
        """

    @abstractmethod
    def encerrar(self) -> None:
        """Libera recursos (ex.: pinos GPIO) ao finalizar a operação."""


@dataclass
class _EventoExtensao:
    timestamp: float
    segundos_solicitados: float


@dataclass
class MockSignalController(SignalController):
    """Controlador de desenvolvimento: não aciona hardware nenhum.

    Registra cada solicitação de extensão em um histórico, para inspeção e
    para os testes de integração da pipeline. Também imprime no console, o
    que já serve como log da PoC ao rodar em vídeo no PC.
    """

    verboso: bool = True
    historico: list[_EventoExtensao] = field(default_factory=list)

    def estender_verde(self, segundos: float) -> None:
        self.historico.append(
            _EventoExtensao(timestamp=time.time(), segundos_solicitados=segundos)
        )
        if self.verboso and segundos > 0:
            print(f"[MockSignalController] extensão solicitada: {segundos:.2f}s")

    def encerrar(self) -> None:
        if self.verboso:
            print("[MockSignalController] encerrado.")

    def maior_extensao_registrada(self) -> float:
        """Utilitário para testes: maior valor de extensão já solicitado."""
        if not self.historico:
            return 0.0
        return max(e.segundos_solicitados for e in self.historico)
