from pydantic import Field
from typing import Any
from enum import Enum
import time

from src.tools.tool_schema import ToolSchema
from src.tools.tool_base import ToolBase
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus


class MathOperation(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    MODULO = "modulo"


class CalculatorSchema(ToolSchema):
    """Schema per il tool calcolatrice che esegue operazioni matematiche di base"""

    # Metadati del tool
    tool_name = "calculator"
    tool_description = "Esegue operazioni matematiche di base tra due numeri (addizione, sottrazione, moltiplicazione, divisione, potenza, modulo)"

    # Parametri del tool
    a: float = Field(..., description="Primo numero dell'operazione")
    b: float = Field(..., description="Secondo numero dell'operazione")
    operation: MathOperation = Field(..., description="Tipo di operazione matematica da eseguire")


class CalculatorTool(ToolBase):
    """Tool per eseguire operazioni matematiche"""

    schema_class = CalculatorSchema

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Esegue l'operazione matematica con parametri standardizzati.

        Args:
            tool_call: ToolCall standardizzato con parametri validati

        Returns:
            ToolResult con il risultato dell'operazione
        """
        start_time = time.time()

        try:
            # Valida i parametri usando Pydantic
            validated_params = self.schema_class(**tool_call.arguments)

            # Esegui l'operazione
            result = self._perform_calculation(
                validated_params.a,
                validated_params.b,
                validated_params.operation
            )

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=result,
                status=ToolStatus.SUCCESS,
                execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=execution_time
            )

    def _perform_calculation(self, a: float, b: float, operation: MathOperation) -> Any:
        """
        Esegue l'operazione matematica specificata.

        Args:
            a: Primo numero
            b: Secondo numero
            operation: Tipo di operazione

        Returns:
            Risultato dell'operazione

        Raises:
            ValueError: Per operazioni non valide
            ZeroDivisionError: Per divisioni per zero
        """
        if operation == MathOperation.ADD:
            result = a + b
        elif operation == MathOperation.SUBTRACT:
            result = a - b
        elif operation == MathOperation.MULTIPLY:
            result = a * b
        elif operation == MathOperation.DIVIDE:
            if b == 0:
                raise ZeroDivisionError("Impossibile dividere per zero")
            result = a / b
        elif operation == MathOperation.POWER:
            result = a ** b
        elif operation == MathOperation.MODULO:
            if b == 0:
                raise ZeroDivisionError("Impossibile calcolare il modulo per zero")
            result = a % b
        else:
            raise ValueError(f"Operazione '{operation}' non supportata")

        # Rimuovi decimali inutili per numeri interi
        if isinstance(result, float) and result == int(result):
            result = int(result)

        return result