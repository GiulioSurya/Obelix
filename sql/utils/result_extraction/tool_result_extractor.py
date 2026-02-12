"""
Modulo per estrazione e conversione di risultati da tool execution.
"""
from src.messages.tool_message import ToolResult
from src.utils.dataframe.converter import DataFrameConverter


def is_sql_executor_tool(tool_result: ToolResult) -> bool:
    """Verifica se il tool_result è di tipo sql_query_executor."""
    return tool_result.tool_name == "sql_query_executor"


def extract_sql_query(tool_result: ToolResult) -> str:
    """
    Estrae la query SQL dal tool_result.

    Returns:
        Query SQL se presente, altrimenti "N/A"
    """
    if tool_result.result is None:
        return "N/A"

    result_data = tool_result.result
    if not isinstance(result_data, dict):
        return "N/A"

    return result_data.get("sql_query", "N/A")


def convert_tool_result_to_dataframe(tool_result: ToolResult) -> object:
    """
    Converte il tool_result in DataFrame o messaggio di errore.

    Returns:
        - DataFrame pandas se conversione riuscita
        - Stringa con errore se presente campo 'error'
        - Stringa "No results" se nessun dato disponibile
    """
    if tool_result.error:
        return f"Error: {tool_result.error}"

    if tool_result.result is None:
        return "No results"

    return DataFrameConverter().from_dict(tool_result.result)


def extract_outputs(
    semantic_search_output: str,
    enhanced_query_content: str,
    result
) -> tuple[str, str, str, object]:
    """
    Estrae le informazioni richieste dal workflow principale.

    Args:
        semantic_search_output: Output catturato della ricerca semantica
        enhanced_query_content: Reasoning del QueryEnhancementAgent
        result: Risultato del SQLGeneratorAgent con tool_results

    Returns:
        Tuple contenente:
        - semantic_search_str: Output formattato della ricerca semantica per colonna
        - reasoning_str: Reasoning del QueryEnhancementAgent
        - sql_query_str: Query SQL eseguita
        - results_df: DataFrame pandas con i risultati della query
    """
    sql_query_str = "N/A"
    results_df = None

    # Early return se nessun tool_result disponibile
    if not result.tool_results:
        return semantic_search_output, enhanced_query_content, sql_query_str, results_df

    # Cerca ed elabora l'ultimo sql_query_executor tool (iterazione al contrario)
    for tool_result in reversed(result.tool_results):
        if not is_sql_executor_tool(tool_result):
            continue

        sql_query_str = extract_sql_query(tool_result)
        results_df = convert_tool_result_to_dataframe(tool_result)
        break

    return semantic_search_output, enhanced_query_content, sql_query_str, results_df
