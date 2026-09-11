"""Modelo matemático da extensão de tempo do semáforo.

Fórmula base (cinemática elementar aplicada ao dimensionamento normativo de
travessia de pedestres — MUTCD/HCM):

    T_extra = clamp( D / V_cadeirante - T_normal ,  0 ,  T_max )

Onde V_cadeirante vem de uma de duas fontes, nessa ordem de preferência:
  1. Velocidade MEDIDA via tracking + homografia, se houver amostras
     suficientes e o valor for fisicamente plausível.
  2. Velocidade FALLBACK conservadora (config), caso contrário.

O princípio de engenharia por trás da escolha entre fontes é fail-safe: na
dúvida, o sistema usa o valor mais conservador (mais lento), que SUPERESTIMA
o tempo necessário. Um erro que dá tempo a mais é seguro; um que dá tempo a
menos não é.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crosswalk.config import SemaphoreConfig

# Faixa fisicamente plausível de velocidade de um cadeirante (m/s).
# Valores fora disso indicam ruído de detecção/tracking, não movimento real.
VELOCIDADE_MIN_PLAUSIVEL_MS = 0.15
VELOCIDADE_MAX_PLAUSIVEL_MS = 2.5

# Frames mínimos rastreados para confiar na velocidade medida em vez do fallback.
FRAMES_MINIMOS_PARA_CONFIAR = 5

# Fator de suavização exponencial (0 < alpha <= 1). Menor = mais suave/lento a reagir.
ALPHA_SUAVIZACAO = 0.3


def clamp(valor: float, minimo: float, maximo: float) -> float:
    """Restringe um valor a um intervalo [minimo, maximo]."""
    return max(minimo, min(valor, maximo))


def calcular_tempo_extra(
    distancia_m: float,
    velocidade_ms: float,
    tempo_normal_s: float,
    tempo_extra_maximo_s: float,
) -> float:
    """Calcula o tempo extra de verde, dada a velocidade do cadeirante.

    T_extra = clamp(D / V - T_normal, 0, T_max)
    """
    if velocidade_ms <= 0:
        raise ValueError("velocidade_ms deve ser positiva")
    if distancia_m < 0:
        raise ValueError("distancia_m não pode ser negativa")

    tempo_necessario = distancia_m / velocidade_ms
    bruto = tempo_necessario - tempo_normal_s
    return clamp(bruto, 0.0, tempo_extra_maximo_s)


@dataclass
class EstimadorVelocidade:
    """Acumula o histórico de posições reais (metros) de UM cadeirante rastreado
    e produz uma estimativa de velocidade suavizada e validada.

    Uma instância por track_id do ByteTrack. Alimentada com a posição do ponto
    de contato (já projetada para o mundo real via homografia) e o TIMESTAMP
    real (em segundos, relógio de parede ou frame_idx/fps para vídeo gravado)
    em que essa observação foi feita.
    """

    velocidade_fallback_ms: float
    _historico_posicoes_m: list[tuple[float, float]] = field(default_factory=list)
    _historico_tempos_s: list[float] = field(default_factory=list)
    _velocidade_suave_ms: float | None = None

    def atualizar(self, posicao_m: tuple[float, float], tempo_s: float) -> None:
        """Registra uma nova observação de posição (metros) no instante tempo_s (segundos)."""
        self._historico_posicoes_m.append(posicao_m)
        self._historico_tempos_s.append(tempo_s)

    @property
    def n_amostras(self) -> int:
        return len(self._historico_posicoes_m)

    def _velocidade_instantanea_ms(self) -> float | None:
        """Velocidade entre as duas últimas observações, ou None se não houver par."""
        if self.n_amostras < 2:
            return None

        (x1, y1), (x2, y2) = (
            self._historico_posicoes_m[-2],
            self._historico_posicoes_m[-1],
        )
        t1, t2 = self._historico_tempos_s[-2], self._historico_tempos_s[-1]

        delta_t = t2 - t1
        if delta_t <= 0:
            return None

        delta_d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return delta_d / delta_t

    def velocidade_atual_ms(self) -> tuple[float, bool]:
        """Retorna (velocidade_a_usar, usou_medicao)."""
        v_inst = self._velocidade_instantanea_ms()

        if v_inst is not None:
            if self._velocidade_suave_ms is None:
                self._velocidade_suave_ms = v_inst
            else:
                self._velocidade_suave_ms = (
                    ALPHA_SUAVIZACAO * v_inst
                    + (1 - ALPHA_SUAVIZACAO) * self._velocidade_suave_ms
                )

        amostras_suficientes = self.n_amostras >= FRAMES_MINIMOS_PARA_CONFIAR
        valor_plausivel = (
            self._velocidade_suave_ms is not None
            and VELOCIDADE_MIN_PLAUSIVEL_MS
            <= self._velocidade_suave_ms
            <= VELOCIDADE_MAX_PLAUSIVEL_MS
        )

        if amostras_suficientes and valor_plausivel:
            return self._velocidade_suave_ms, True

        return self.velocidade_fallback_ms, False


def decidir_extensao(
    velocidades_ms: list[float],
    semaphore_config: SemaphoreConfig,
    distancia_m: float,
) -> float:
    """Decide o tempo extra final considerando TODOS os cadeirantes na faixa."""
    if not velocidades_ms:
        return 0.0

    tempos = [
        calcular_tempo_extra(
            distancia_m=distancia_m,
            velocidade_ms=v,
            tempo_normal_s=semaphore_config.tempo_verde_normal_s,
            tempo_extra_maximo_s=semaphore_config.tempo_extra_maximo_s,
        )
        for v in velocidades_ms
    ]
    return max(tempos)
