from src.providers import Providers



class GlobalConfig:
    _instance = None
    _current_provider = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_provider(self, provider: Providers):
        self._current_provider = provider

    def get_current_provider(self) -> Providers:
        if self._current_provider is None:
            raise ValueError("Provider non impostato. Usa blackboard.set_provider()")
        return self._current_provider


