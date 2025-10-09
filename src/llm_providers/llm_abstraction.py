# src/llm_providers/abstract_llm_provider.py
from abc import ABC, abstractmethod
from typing import List

from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantMessage
from src.tools.tool_base import ToolBase
from abc import ABCMeta, abstractmethod, ABC
import threading

class SingletonMeta(ABCMeta):
    """
    Metaclasse singleton thread-safe per garantire una sola istanza
    di ogni classe LLM nel sistema multi-agent.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Pattern di double-checked locking
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractLLMProvider(ABC, metaclass=SingletonMeta):
    """
    Classe base astratta per i provider LLM.

    Ogni provider implementa internamente la propria logica di conversione
    (es. usando strategy pattern, metodi diretti, ecc.)

    L'unica interfaccia pubblica obbligatoria è invoke().
    """

    @abstractmethod
    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Chiama il modello LLM con messaggi e tool standardizzati

        Args:
            messages: Lista di messaggi in formato StandardMessage
            tools: Lista di tool disponibili

        Returns:
            AssistantMessage con la risposta del modello
        """
        pass