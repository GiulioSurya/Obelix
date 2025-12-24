# src/agents/base_agent.py
import asyncio
from typing import List, Optional, Dict, Any, Union, Callable
from src.logging_config import get_logger
from src.base_agent.agent_schema import AgentSchema, ToolDescription
from src.messages.system_message import SystemMessage
from src.messages.human_message import HumanMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage, ToolCall, ToolResult, ToolStatus
from src.base_agent.middleware import AgentEvent, Middleware, MiddlewareContext
from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantResponse
from src.messages.usage import AgentUsage
from src.tools.tool_base import ToolBase
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.config import GlobalConfig

logger = get_logger(__name__)

class BaseAgent:
    def __init__(self,
                 system_message: str,
                 provider: Optional[AbstractLLMProvider] = None,
                 agent_name: str = None,
                 description: str = None,
                 agent_comment: bool = True):
        self.system_message = SystemMessage(content=system_message)
        self.agent_comment = agent_comment

        self.provider = provider or GlobalConfig().get_current_provider_instance()

        self.agent_name = agent_name or self.__class__.__name__
        self.description = description if description is not None else "AI assistant for specialized task execution"
        self.registered_tools: List[ToolBase] = []
        self.conversation_history: List[StandardMessage] = [self.system_message]
        self.agent_usage = AgentUsage(model_id=self.provider.model_id)

        # Middleware system
        self._middlewares: Dict[AgentEvent, List[Middleware]] = {
            event: [] for event in AgentEvent
        }

    def register_tool(self, tool: ToolBase):
        """
        Registra un tool per l'agent

        Args:
            tool: Tool da registrare (ToolBase o MCPTool)

        Raises:
            RuntimeError: Se il tool MCP non è connesso
        """
        # Lazy import di MCPTool per evitare dipendenze Windows su Linux/Docker
        try:
            from src.tools.mcp.mcp_tool import MCPTool
            # Se è un MCPTool, verifica che il manager sia connesso
            if isinstance(tool, MCPTool):
                if not tool.manager.is_connected():
                    logger.error(f"Agent {self.agent_name}: tentativo registrazione MCP Tool {tool.tool_name} fallito - manager non connesso")
                    raise RuntimeError(f"MCP Tool {tool.tool_name} non è connesso. "
                                       f"Il manager deve essere connesso prima della registrazione.")


        except ImportError:
            # MCPTool non disponibile (es. pywin32 mancante su Windows o ambiente Docker)
            logger.debug("MCP tools non disponibili (import libreria mcp fallito)")

        # Registra il tool
        if tool not in self.registered_tools:
            self.registered_tools.append(tool)
            # Accesso diretto a tool.tool_name (popolato dal decoratore @tool o da MCPTool)
            tool_name = getattr(tool, 'tool_name', None) or tool.__class__.__name__
            logger.info(f"Agent {self.agent_name}: tool '{tool_name}' registrato")

    def on(self, event: AgentEvent) -> Middleware:
        """
        API fluente per registrare middleware.

        Esempi:
            agent.on(AgentEvent.ON_TOOL_ERROR).when(...).inject(...)
            agent.on(AgentEvent.AFTER_LLM_CALL).transform(...)

        Args:
            event: L'evento su cui registrare il middleware

        Returns:
            Middleware: Oggetto middleware per method chaining
        """
        middleware = Middleware(event)
        self._middlewares[event].append(middleware)
        return middleware

    async def _trigger_middleware(
        self,
        event: AgentEvent,
        current_value: Any = None,
        **ctx_kwargs
    ) -> Any:
        """
        Esegue tutti i middleware registrati per un evento.

        Args:
            event: L'evento da triggerare
            current_value: Valore corrente da passare ai middleware
            **ctx_kwargs: Parametri aggiuntivi per MiddlewareContext

        Returns:
            Valore trasformato dai middleware
        """
        ctx = MiddlewareContext(
            event=event,
            agent=self,
            **ctx_kwargs
        )

        result = current_value
        for middleware in self._middlewares[event]:
            result = await middleware.execute(ctx, result)

        return result

    def get_agent_schema(self) -> AgentSchema:
        tool_descriptions = []

        for tool in self.registered_tools:
            try:
                schema = tool.create_schema()
                parameter_keys = []
                if hasattr(schema, 'inputSchema') and schema.inputSchema:
                    properties = schema.inputSchema.get('properties', {})
                    parameter_keys = list(properties.keys())

                tool_descriptions.append(ToolDescription(
                    name=schema.name,
                    description=schema.description or f"Tool {schema.name}",
                    parameters=parameter_keys
                ))
            except Exception as e:
                tool_name = getattr(tool, 'tool_name', tool.__class__.__name__)
                tool_descriptions.append(ToolDescription(
                    name=tool_name,
                    description=f"Tool {tool_name} (schema non disponibile: {e})",
                    parameters=[]
                ))

        # Genera output schema dinamico dai tool
        dynamic_output_schema = self.generate_unified_output_schema()

        return AgentSchema(
            name=self.agent_name,
            description=self.description,
            capabilities={
                "tools": len(tool_descriptions) > 0,
                "available_tools": [tool.model_dump() for tool in tool_descriptions]
            },
            output_schema=dynamic_output_schema
        )

    def generate_unified_output_schema(self) -> Dict[str, Any]:
        """Schema generico che include outputs di tutti i tool"""
        tool_outputs = {}

        for tool in self.registered_tools:
            try:
                tool_schema = tool.create_schema()
                if hasattr(tool_schema, 'outputSchema') and tool_schema.outputSchema:
                    tool_outputs[tool_schema.name] = tool_schema.outputSchema
            except Exception as e:
                print(f"Impossibile creare schema per il tool: {e}")
                # WARNING: operazione fallita ma app continua con fallback
                logger.warning(f"Impossibile creare schema per il tool: {e}")
                continue

        properties = {
            "result": {"type": "string", "description": "Risultato principale"}
        }

        # Aggiungi tool_outputs solo se ci sono tool
        if tool_outputs:
            properties["tool_outputs"] = {
                "type": "object",
                "properties": tool_outputs,
                "description": "Output specifici dei tool utilizzati"
            }

        return {
            "type": "object",
            "properties": properties
        }

    async def execute_query_async(
        self,
        query: Union[str, List[StandardMessage]],
        max_attempts: int = 3,
        max_iterations: int = 5
    ) -> AssistantResponse:
        """
        Esegue query in modo asincrono (per FastAPI).

        Versione async-native che usa l'event loop esistente.
        Include retry logic per errori fatali (crash LLM, network timeout) e
        validazione input.

        Args:
            query: Query da eseguire. Può essere:
                   - str: stringa convertita automaticamente in HumanMessage
                   - List[StandardMessage]: lista di messaggi (deve contenere esattamente 1 HumanMessage)
            max_attempts: Numero massimo di tentativi in caso di crash/errore fatale (default: 3)
            max_iterations: Numero massimo di iterazioni del loop tool-call (default: 5)

        Returns:
            AssistantResponse: Risposta strutturata dell'agent

        Raises:
            ValueError: Se la lista di messaggi non contiene esattamente 1 HumanMessage
            TypeError: Se query non è str o List[StandardMessage]
            RuntimeError: Se tutti i tentativi falliscono
        """
        # 1. Validazione input
        self._validate_query_input(query)

        # 2. Backup conversation history per retry puliti
        original_history = self.conversation_history.copy()

        # 3. Retry loop per errori fatali
        for attempt in range(max_attempts):
            try:
                return await self._async_execute_query(query, max_iterations)
            except (ValueError, TypeError):
                # Errori di validazione non vanno ritentati (già validato)
                raise
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Esecuzione fallita dopo {max_attempts} tentativi. "
                        f"Ultimo errore: {e}"
                    )
                # Ripristina history per retry pulito
                self.conversation_history = original_history.copy()
                #print(f"Tentativo {attempt + 1} fallito: {e}. Riprovo...")
                # WARNING: retry in corso - situazione anomala ma gestita
                logger.warning(f"Tentativo {attempt + 1}/{max_attempts} fallito: {e}. Riprovo...")

        raise RuntimeError("Esecuzione fallita per motivi sconosciuti")

    def execute_query(
        self,
        query: Union[str, List[StandardMessage]],
        max_attempts: int = 3,
        max_iterations: int = 3
    ) -> AssistantResponse:
        """
        Esegue query in modo sincrono (per CLI).

        Wrapper sync che crea un nuovo event loop per eseguire la versione async.
        Include retry logic per errori fatali e validazione input.

        Args:
            query: Query da eseguire. Può essere:
                   - str: stringa convertita automaticamente in HumanMessage
                   - List[StandardMessage]: lista di messaggi (deve contenere esattamente 1 HumanMessage)
            max_attempts: Numero massimo di tentativi in caso di crash/errore fatale (default: 3)
            max_iterations: Numero massimo di iterazioni del loop tool-call (default: 5)

        Returns:
            AssistantResponse: Risposta strutturata dell'agent

        Raises:
            ValueError: Se la lista di messaggi non contiene esattamente 1 HumanMessage
            TypeError: Se query non è str o List[StandardMessage]
            RuntimeError: Se tutti i tentativi falliscono
        """
        return asyncio.run(self.execute_query_async(query, max_attempts, max_iterations))

    def _validate_query_input(self, query: Union[str, List[StandardMessage]]) -> None:
        """
        Valida l'input della query.

        Args:
            query: Query da validare

        Raises:
            ValueError: Se la lista di messaggi non contiene esattamente 1 HumanMessage
            TypeError: Se query non è str o List[StandardMessage]
        """
        if isinstance(query, list):
            human_messages = [msg for msg in query if isinstance(msg, HumanMessage)]
            if len(human_messages) != 1:
                raise ValueError(
                    f"La lista di messaggi deve contenere esattamente 1 HumanMessage, "
                    f"trovati {len(human_messages)}"
                )
        elif not isinstance(query, str):
            raise TypeError(
                f"query deve essere str o List[StandardMessage], "
                f"ricevuto {type(query).__name__}"
            )

    async def _async_execute_query(self, query: Union[str, List[StandardMessage]], max_iterations: int = 5) -> AssistantResponse:
        """
        Esecuzione asincrona della query (cuore dell'esecuzione)

        Args:
            query: Query da eseguire. Può essere:
                   - str: stringa convertita automaticamente in HumanMessage
                   - List[StandardMessage]: lista di messaggi (deve contenere esattamente 1 HumanMessage)
            max_iterations: Numero massimo di iterazioni del loop tool-call (default: 5)

        Returns:
            AssistantResponse: Risposta strutturata dell'agent

        Raises:
            ValueError: Se la lista di messaggi non contiene esattamente 1 HumanMessage
        """
        collected_tool_results = []
        execution_error = None

        # Gestisci input: stringa o lista di messaggi
        if isinstance(query, str):
            # Caso 1: stringa → crea HumanMessage
            user_message = HumanMessage(content=query)
            self.conversation_history.append(user_message)
        elif isinstance(query, list):
            # Caso 2: lista di messaggi → valida e aggiungi
            # Conta HumanMessage nella lista
            human_messages = [msg for msg in query if isinstance(msg, HumanMessage)]

            if len(human_messages) != 1:
                raise ValueError(
                    f"La lista di messaggi deve contenere esattamente 1 HumanMessage, "
                    f"trovati {len(human_messages)}"
                )

            # Aggiungi tutti i messaggi alla conversation history
            self.conversation_history.extend(query)
        else:
            raise TypeError(
                f"query deve essere str o List[StandardMessage], "
                f"ricevuto {type(query).__name__}"
            )

        # >>> MIDDLEWARE: ON_QUERY_START <<<
        await self._trigger_middleware(AgentEvent.ON_QUERY_START, iteration=0)

        # Loop multi-turn per gestire tool calls
        for iteration in range(1, max_iterations + 1):
            # >>> MIDDLEWARE: BEFORE_LLM_CALL <<<
            await self._trigger_middleware(AgentEvent.BEFORE_LLM_CALL, iteration=iteration)

            assistant_msg = self.provider.invoke(self.conversation_history, self.registered_tools)

            # >>> MIDDLEWARE: AFTER_LLM_CALL <<< (può trasformare assistant_msg)
            assistant_msg = await self._trigger_middleware(
                AgentEvent.AFTER_LLM_CALL,
                current_value=assistant_msg,
                iteration=iteration,
                assistant_message=assistant_msg
            )

            # Traccia usage della chiamata LLM (se disponibile)
            if assistant_msg.usage:
                self.agent_usage.add_usage(assistant_msg.usage)

            # Caso 1: Risposta vuota → uscita valida del loop
            if not assistant_msg.tool_calls and not assistant_msg.content:
                #print("[INFO] Agent ha completato l'esecuzione senza risposta testuale finale.")
                # INFO: evento normale dell'applicazione
                logger.info("Agent ha completato l'esecuzione senza risposta testuale finale")
                self.conversation_history.append(assistant_msg)
                response = self._build_final_response(
                    assistant_msg,
                    collected_tool_results,
                    execution_error
                )
                # >>> MIDDLEWARE: ON_QUERY_END <<<
                await self._trigger_middleware(AgentEvent.ON_QUERY_END, iteration=iteration)
                return response

            # Caso 2: Tool calls → elabora
            if assistant_msg.tool_calls:
                execution_error = await self._process_tool_calls(
                    assistant_msg, collected_tool_results, execution_error, iteration
                )

                # Se agent_comment=False e nessun errore, esci senza far commentare LLM
                if not self.agent_comment and execution_error is None:
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._trigger_middleware(AgentEvent.ON_QUERY_END, iteration=iteration)
                    return response

                continue

            # Caso 3: Contenuto testuale → risposta finale
            response = self._build_final_response(
                assistant_msg, collected_tool_results, execution_error
            )
            await self._trigger_middleware(AgentEvent.ON_QUERY_END, iteration=iteration)
            return response

        # >>> MIDDLEWARE: ON_MAX_ITERATIONS <<<
        await self._trigger_middleware(AgentEvent.ON_MAX_ITERATIONS, iteration=max_iterations)
        return self._build_timeout_response(
            max_iterations, collected_tool_results, execution_error
        )

    async def _process_tool_calls(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[Dict[str, Any]],
        execution_error: Optional[str],
        iteration: int
    ) -> Optional[str]:
        """
        Elabora le tool calls dell'assistant.

        Integra il sistema middleware per permettere estensioni senza override:
        - BEFORE_TOOL_EXECUTION: prima di execute(), può modificare ToolCall
        - AFTER_TOOL_EXECUTION: dopo execute(), può modificare ToolResult
        - ON_TOOL_ERROR: quando un tool ritorna errore
        - BEFORE_HISTORY_UPDATE: prima di aggiungere a history

        Args:
            assistant_msg: Messaggio dell'assistant con le tool calls
            collected_tool_results: Lista per raccogliere i risultati
            execution_error: Errore di esecuzione corrente (se presente)
            iteration: Numero iterazione corrente

        Returns:
            Optional[str]: Errore di esecuzione aggiornato (se presente)
        """
        # print("\n[DEBUG] Tool calls ricevute dall'LLM:")
        # from pprint import pprint
        # pprint(assistant_msg.tool_calls[0].arguments, indent=2)
        # DEBUG: dettagli utili durante sviluppo
        logger.debug(f"Tool calls ricevute: {[tc.name for tc in assistant_msg.tool_calls]}")
        logger.debug(f"Primo tool arguments: {assistant_msg.tool_calls[0].arguments}")

        tool_results = []

        for call in assistant_msg.tool_calls:
            # >>> MIDDLEWARE: BEFORE_TOOL_EXECUTION <<< (può trasformare call)
            call = await self._trigger_middleware(
                AgentEvent.BEFORE_TOOL_EXECUTION,
                current_value=call,
                iteration=iteration,
                tool_call=call
            )

            # Esecuzione tool
            result = await self._async_execute_tool(call)
            if not result:
                # Crea un ToolResult di fallback per evitare errori OCI
                result = ToolResult(
                    tool_name=call.name,
                    tool_call_id=call.id,
                    result=None,
                    status=ToolStatus.ERROR,
                    error=f"Tool {call.name} not found or not executable"
                )

            # >>> MIDDLEWARE: AFTER_TOOL_EXECUTION <<< (può trasformare result)
            result = await self._trigger_middleware(
                AgentEvent.AFTER_TOOL_EXECUTION,
                current_value=result,
                iteration=iteration,
                tool_result=result
            )

            # >>> MIDDLEWARE: ON_TOOL_ERROR <<< (solo se errore)
            if result.status == ToolStatus.ERROR:
                await self._trigger_middleware(
                    AgentEvent.ON_TOOL_ERROR,
                    iteration=iteration,
                    tool_result=result,
                    error=result.error
                )

            # Log tool execution result
            #print(f"Executed {call.name}, result: {result.model_dump_json(indent=2)}")
            # DEBUG: risultato esecuzione tool (dettaglio sviluppo)
            logger.debug(f"Executed {call.name}, status={result.status}")

            tool_results.append(result)

            # Raccogli dati per la risposta finale
            tool_summary = self._create_tool_summary(result)
            collected_tool_results.append(tool_summary)

            # Traccia il primo errore
            if result.error and not execution_error:
                execution_error = f"Errore in {result.tool_name}: {result.error}"

            # >>> MIDDLEWARE: BEFORE_HISTORY_UPDATE <<<
            await self._trigger_middleware(
                AgentEvent.BEFORE_HISTORY_UPDATE,
                iteration=iteration,
                tool_result=result
            )

        # Aggiungi messaggi alla conversazione
        tool_message = ToolMessage(tool_results=tool_results)
        self.conversation_history.extend([assistant_msg, tool_message])

        # Se l'ultimo tool ha successo, resetta l'errore precedente
        # Permette uscita corretta con agent_comment=False dopo un retry riuscito
        if tool_results and tool_results[-1].status == ToolStatus.SUCCESS:
            execution_error = None

        return execution_error

    def _create_tool_summary(self, result: ToolResult) -> Dict[str, Any]:
        """
        Crea un summary del risultato del tool

        Args:
            result: Risultato del tool

        Returns:
            Dict contenente le informazioni essenziali del tool
        """
        tool_summary = {"tool_name": result.tool_name}

        if result.status == "success" and result.result is not None:
            tool_summary["result"] = result.result

        if result.error:
            tool_summary["error"] = result.error

        return tool_summary

    def _build_final_response(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[Dict[str, Any]],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        """
        Costruisce la risposta finale dell'agent

        Args:
            assistant_msg: Messaggio finale dell'assistant
            collected_tool_results: Risultati dei tool eseguiti
            execution_error: Errore di esecuzione (se presente)

        Returns:
            AssistantResponse strutturata
        """
        # Fallback per content vuoto
        final_content = assistant_msg.content if assistant_msg.content else "Esecuzione completata."

        #print(f"Assistant response: {final_content}")
        # INFO: operazione di business completata
        logger.info(f"Assistant response generata per agent {self.agent_name}")
        self.conversation_history.append(assistant_msg)

        return AssistantResponse(
            agent_name=self.agent_name,
            content=final_content,
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error
        )

    def _build_timeout_response(
        self,
        max_iterations: int,
        collected_tool_results: List[Dict[str, Any]],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        """
        Costruisce la risposta quando si raggiunge il limite di iterazioni

        Args:
            max_iterations: Numero massimo di iterazioni raggiunto
            collected_tool_results: Risultati dei tool eseguiti
            execution_error: Errore di esecuzione (se presente)

        Returns:
            AssistantResponse con warning per timeout
        """
        warning_msg = f"Raggiunto il limite di {max_iterations} iterazioni. Esecuzione interrotta."
        print(warning_msg)
        # WARNING: limite raggiunto, situazione anomala
        logger.warning(warning_msg)

        return AssistantResponse(
            agent_name=self.agent_name,
            content=f"Esecuzione interrotta dopo {max_iterations} iterazioni.",
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error or warning_msg
        )

    async def _async_execute_tool(self, tool_call: ToolCall) -> Optional[ToolResult]:
        """
        Helper per l'esecuzione asincrona dei tool

        Args:
            tool_call: Chiamata al tool da eseguire

        Returns:
            ToolResult: Risultato dell'esecuzione del tool, None se tool non trovato
        """

        for tool in self.registered_tools:
            # Accesso diretto a tool.tool_name (decoratore @tool o MCPTool)
            tool_name = getattr(tool, 'tool_name', None)
            if tool_name == tool_call.name:
                try:
                    result = await tool.execute(tool_call)
                    return result
                except Exception as e:
                    # Crea un ToolResult con errore invece di fallire completamente
                    return ToolResult(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        result=None,
                        status="error",
                        error=f"Errore esecuzione tool: {e}"
                    )

        # print(f"[ERROR] Tool {tool_call.name} non trovato tra i tool registrati")
        # print(f"[ERROR] Registered tools: {[getattr(t, 'tool_name', t.__class__.__name__) for t in self.registered_tools]}")
        # ERROR: operazione fallita, tool non trovato
        logger.error(f"Tool {tool_call.name} non trovato tra i tool registrati")
        logger.error(f"Registered tools: {[getattr(t, 'tool_name', t.__class__.__name__) for t in self.registered_tools]}")
        return None

    def get_conversation_history(self) -> List[StandardMessage]:
        """
        Restituisce la cronologia della conversazione

        Returns:
            List[StandardMessage]: Lista dei messaggi della conversazione
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self, keep_system_message: bool = True):
        """
        Pulisce la cronologia della conversazione

        Args:
            keep_system_message: Se mantenere il messaggio di sistema
        """
        if keep_system_message:
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []
