"""
Sistema Hook per BaseAgent.

Fornisce un'API fluente per intercettare eventi del ciclo di vita dell'agent,
iniettare messaggi nella conversation history e trasformare risultati.

Esempio:
    agent.on(AgentEvent.ON_TOOL_ERROR) \
        .when(lambda ctx: "invalid identifier" in ctx.error) \
        .inject(lambda ctx: HumanMessage(content="Schema: ..."))
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any, List, Union, TYPE_CHECKING
import asyncio

from src.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.base_agent.base_agent import BaseAgent
    from src.messages.tool_message import ToolCall, ToolResult
    from src.messages.assistant_message import AssistantMessage
    from src.messages.standard_message import StandardMessage


class AgentEvent(str, Enum):
    """Eventi del ciclo di vita dell'agent"""

    # === Lifecycle ===
    ON_QUERY_START = "on_query_start"
    """Prima di iniziare l'esecuzione della query"""

    ON_QUERY_END = "on_query_end"
    """Fine esecuzione (successo o errore)"""

    # === LLM ===
    BEFORE_LLM_CALL = "before_llm_call"
    """Prima di chiamare provider.invoke()"""

    AFTER_LLM_CALL = "after_llm_call"
    """Dopo la risposta LLM (può trasformare AssistantMessage)"""

    # === Tool Calls ===
    BEFORE_TOOL_EXECUTION = "before_tool_execution"
    """Prima di eseguire tool.execute() (può trasformare ToolCall)"""

    AFTER_TOOL_EXECUTION = "after_tool_execution"
    """Dopo l'esecuzione del tool (può trasformare ToolResult)"""

    # === Condizionali ===
    ON_TOOL_ERROR = "on_tool_error"
    """Quando un tool ritorna errore (status == ERROR)"""

    ON_MAX_ITERATIONS = "on_max_iterations"
    """Quando viene raggiunto il limite di iterazioni"""


@dataclass
class HookContext:
    """
    Contesto ricco passato ai hook.

    Contiene informazioni sullo stato corrente dell'agent
    e permette l'accesso alla conversation history.
    """
    event: AgentEvent
    agent: 'BaseAgent'
    iteration: int = 0
    tool_call: Optional['ToolCall'] = None
    tool_result: Optional['ToolResult'] = None
    assistant_message: Optional['AssistantMessage'] = None
    error: Optional[str] = None

    @property
    def conversation_history(self) -> List['StandardMessage']:
        """Accesso diretto alla conversation history dell'agent"""
        return self.agent.conversation_history


class Hook:
    """
    Hook con API fluente per intercettare eventi dell'agent.

    Supporta:
    - Condizioni di attivazione (.when())
    - Iniezione messaggi (.inject(), .inject_at())
    - Trasformazione valori (.transform())
    - Azioni generiche (.do())

    Esempio:
        agent.on(AgentEvent.AFTER_TOOL_EXECUTION) \
            .when(lambda ctx: ctx.tool_result.error) \
            .transform(lambda result, ctx: enrich_error(result))
    """

    def __init__(self, event: AgentEvent):
        self.event = event
        self._condition: Optional[Callable[[HookContext], bool]] = None
        self._actions: List[Callable] = []
        logger.debug(f"Hook creato per evento: {event.value}")

    def when(self, condition: Callable[[HookContext], bool]) -> 'Hook':
        """
        Imposta la condizione per attivare l'hook.

        Args:
            condition: Funzione che riceve HookContext e ritorna bool.
                      Può essere sync o async.

        Returns:
            self per method chaining
        """
        self._condition = condition
        condition_name = getattr(condition, '__name__', str(condition))
        logger.debug(f"Hook [{self.event.value}] - condizione impostata: {condition_name}")
        return self

    def inject(
        self,
        message_factory: Callable[[HookContext], 'StandardMessage']
    ) -> 'Hook':
        """
        Inietta un messaggio alla fine della conversation history (append).

        Args:
            message_factory: Funzione che riceve HookContext e ritorna
                           il messaggio da iniettare

        Returns:
            self per method chaining
        """
        factory_name = getattr(message_factory, '__name__', str(message_factory))
        logger.debug(f"Hook [{self.event.value}] - registrata inject action: {factory_name}")

        def action(ctx: HookContext, current: Any) -> Any:
            message = message_factory(ctx)
            logger.debug(f"Hook [{ctx.event.value}] - inject eseguita, messaggio tipo: {type(message).__name__}")
            ctx.agent.conversation_history.append(message)
            return current

        self._actions.append(action)
        return self

    def inject_at(
        self,
        position: Union[int, Callable[[HookContext], int]],
        message_factory: Callable[[HookContext], 'StandardMessage']
    ) -> 'Hook':
        """
        Inietta un messaggio a una posizione specifica nella conversation history.

        Args:
            position: Indice dove inserire, oppure funzione che calcola l'indice.
                     Valori negativi sono relativi alla fine (-1 = prima dell'ultimo)
            message_factory: Funzione che ritorna il messaggio da iniettare

        Returns:
            self per method chaining
        """
        factory_name = getattr(message_factory, '__name__', str(message_factory))
        pos_desc = position if isinstance(position, int) else getattr(position, '__name__', 'dynamic')
        logger.debug(f"Hook [{self.event.value}] - registrata inject_at action: {factory_name} @ pos={pos_desc}")

        def action(ctx: HookContext, current: Any) -> Any:
            message = message_factory(ctx)
            pos = position(ctx) if callable(position) else position
            logger.debug(f"Hook [{ctx.event.value}] - inject_at eseguita @ pos={pos}, messaggio tipo: {type(message).__name__}")
            ctx.agent.conversation_history.insert(pos, message)
            return current

        self._actions.append(action)
        return self

    def do(self, action: Callable[[HookContext], Any]) -> 'Hook':
        """
        Esegue un'azione generica (logging, side effects, etc.).

        L'azione non modifica il valore corrente del chain.

        Args:
            action: Funzione che riceve HookContext. Può essere sync o async.

        Returns:
            self per method chaining
        """
        action_name = getattr(action, '__name__', str(action))
        logger.debug(f"Hook [{self.event.value}] - registrata do action: {action_name}")

        def wrapped(ctx: HookContext, current: Any) -> Any:
            logger.debug(f"Hook [{ctx.event.value}] - do action eseguita: {action_name}")
            action(ctx)
            return current

        self._actions.append(wrapped)
        return self

    def transform(
        self,
        transformer: Callable[[Any, HookContext], Any]
    ) -> 'Hook':
        """
        Trasforma il valore corrente (ToolResult, AssistantMessage, ToolCall).

        Utile con eventi come AFTER_TOOL_EXECUTION o AFTER_LLM_CALL.

        Args:
            transformer: Funzione (current_value, ctx) -> new_value.
                        Può essere sync o async.

        Returns:
            self per method chaining
        """
        transformer_name = getattr(transformer, '__name__', str(transformer))
        logger.debug(f"Hook [{self.event.value}] - registrata transform action: {transformer_name}")

        def action(ctx: HookContext, current: Any) -> Any:
            logger.debug(f"Hook [{ctx.event.value}] - transform eseguita: {transformer_name}, input tipo: {type(current).__name__}")
            result = transformer(current, ctx)
            logger.debug(f"Hook [{ctx.event.value}] - transform completata, output tipo: {type(result).__name__}")
            return result

        self._actions.append(action)
        return self

    async def execute(self, ctx: HookContext, current_value: Any = None) -> Any:
        """
        Esegue l'hook se la condizione è soddisfatta.

        Args:
            ctx: Contesto del hook
            current_value: Valore corrente da passare alle trasformazioni

        Returns:
            Valore trasformato o current_value se nessuna trasformazione
        """
        logger.debug(f"Hook [{self.event.value}] - execute invocato, iterazione={ctx.iteration}")

        # Check condizione
        if self._condition is not None:
            condition_name = getattr(self._condition, '__name__', 'anonymous')
            logger.debug(f"Hook [{self.event.value}] - valutazione condizione: {condition_name}")
            cond_result = self._condition(ctx)
            if asyncio.iscoroutine(cond_result):
                cond_result = await cond_result
            if not cond_result:
                logger.debug(f"Hook [{self.event.value}] - condizione NON soddisfatta, skip")
                return current_value
            logger.debug(f"Hook [{self.event.value}] - condizione SODDISFATTA, eseguo {len(self._actions)} azioni")
        else:
            logger.debug(f"Hook [{self.event.value}] - nessuna condizione, eseguo {len(self._actions)} azioni")

        # Esegue tutte le azioni in sequenza
        result = current_value
        for i, action in enumerate(self._actions):
            logger.debug(f"Hook [{self.event.value}] - esecuzione azione {i+1}/{len(self._actions)}")
            action_result = action(ctx, result)
            if asyncio.iscoroutine(action_result):
                action_result = await action_result
            if action_result is not None:
                result = action_result

        logger.debug(f"Hook [{self.event.value}] - execute completato")
        return result
