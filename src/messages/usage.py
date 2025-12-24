# src/messages/usage.py
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Modello per tracciare l'utilizzo dei token LLM"""
    input_tokens: int = Field(..., ge=0, description="Numero di token in input (prompt)")
    output_tokens: int = Field(..., ge=0, description="Numero di token in output (completion)")
    total_tokens: int = Field(..., ge=0, description="Numero totale di token utilizzati")

    class Config:
        # Validazione strict per assignment
        validate_assignment = True


class AgentUsage(BaseModel):
    """Accumula l'utilizzo totale dei token per un agente"""
    model_id: str = Field(..., description="ID del modello LLM")
    total_input_tokens: int = Field(default=0, ge=0, description="Somma di tutti i token in input")
    total_output_tokens: int = Field(default=0, ge=0, description="Somma di tutti i token in output")
    total_tokens: int = Field(default=0, ge=0, description="Somma totale di tutti i token")
    call_count: int = Field(default=0, ge=0, description="Numero di chiamate LLM effettuate")

    class Config:
        # Validazione strict per assignment
        validate_assignment = True

    def add_usage(self, usage: Usage) -> None:
        """
        Somma i token di una nuova chiamata LLM all'accumulo totale

        Args:
            usage: Oggetto Usage con i token della singola chiamata
        """
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1

    def reset(self) -> None:
        """Reset dei contatori a zero"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
