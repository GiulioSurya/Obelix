"""
Table Generator Module
======================

Genera tabelle HTML Plotly da DataFrame pandas.
Modulo standalone riutilizzabile in diversi contesti.
"""

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def generate_data_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    """
    Genera una tabella HTML Plotly dal DataFrame.

    Args:
        df: DataFrame pandas da visualizzare
        max_rows: Numero massimo di righe da mostrare (default 500)

    Returns:
        HTML string della tabella Plotly, stringa vuota se errore
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    if df is None or df.empty:
        return ""

    display_df = df.copy()
    is_truncated = len(display_df) > max_rows

    if is_truncated:
        display_df = display_df.head(max_rows)

    # Formatta valori numerici
    formatted_values = []
    for col in display_df.columns:
        col_values = []
        for val in display_df[col]:
            if pd.isna(val):
                col_values.append("N/A")
            elif isinstance(val, float) and val != int(val):
                col_values.append(f"{val:,.2f}")
            elif isinstance(val, (int, float)):
                col_values.append(f"{int(val):,}")
            else:
                col_values.append(str(val))
        formatted_values.append(col_values)

    # Colori tabella
    header_color = '#2c3e50'
    row_colors = ['#ecf0f1' if i % 2 == 0 else 'white' for i in range(len(display_df))]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in display_df.columns],
            fill_color=header_color,
            font=dict(color='white', size=14, family="Arial"),
            align='left',
            height=40,
            line=dict(color='#34495e', width=2)
        ),
        cells=dict(
            values=formatted_values,
            fill_color=[row_colors],
            font=dict(color='#2c3e50', size=12, family="Arial"),
            align='left',
            height=30,
            line=dict(color='#bdc3c7', width=1)
        )
    )])

    # Titolo tabella
    table_title = "<b>Complete Dataset Table</b>"
    if is_truncated:
        table_title += f"<br><sub>Showing first {max_rows} of {len(df):,} rows</sub>"
    else:
        table_title += f"<br><sub>Total rows: {len(df):,}</sub>"

    fig.update_layout(
        title={
            'text': table_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        margin=dict(t=80, l=20, r=20, b=20),
        height=min(600, 80 + len(display_df) * 30 + 50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig.to_html(
        include_plotlyjs='cdn',
        include_mathjax='cdn',
        full_html=False
    )


def should_skip_charts(metadata: dict) -> bool:
    """
    Determina se i chart dovrebbero essere saltati in base ai metadata del DataFrame.

    Salta la generazione chart quando:
    - DataFrame ha solo 1 riga
    - DataFrame ha solo 1 colonna
    - DataFrame ha sia 1 riga che 1 colonna

    Args:
        metadata: Dizionario metadata da DataFrameAnalyzer.analyze()

    Returns:
        True se i chart dovrebbero essere saltati, False altrimenti
    """
    n_rows = metadata.get("n_rows", 0)
    n_columns = metadata.get("n_columns", 0)

    skip = n_rows <= 1 or n_columns <= 1

    if skip:
        logger.debug(
            f"Chart generation skipped: n_rows={n_rows}, n_columns={n_columns}. "
            f"Reason: {'single row' if n_rows <= 1 else ''}"
            f"{' and ' if n_rows <= 1 and n_columns <= 1 else ''}"
            f"{'single column' if n_columns <= 1 else ''}"
        )

    return skip
