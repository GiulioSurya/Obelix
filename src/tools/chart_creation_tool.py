"""
Chart Creation Tool - Simplified Version
=========================================

Tool semplificato per creare visualizzazioni grafiche da DataFrame pandas usando Plotly.
Il DataFrame viene iniettato al momento della creazione del tool e non è visibile all'LLM.
"""

from pydantic import Field
from typing import Optional, Literal, List, Any
import pandas as pd
from pathlib import Path

from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase


# Palette colori multicolor hardcoded (Plotly Bold palette)
MULTICOLOR_PALETTE = [
    '#7F3C8D', '#11A579', '#3969AC', '#F2B701', '#E73F74',
    '#80BA5A', '#E68310', '#008695', '#CF1C90', '#f97b72',
    '#4b4b8f', '#A5AA99'
]


@tool(
    name="create_chart",
    description=(
        "Crea visualizzazioni grafiche dai dati analizzati. Usare massimo 3-4 volte per query. "
        "Supporta 8 tipi di grafici: line, bar, scatter, histogram, pie, box, sunburst, treemap."
    )
)
class ChartCreationTool(ToolBase):
    """
    Tool semplificato per creare grafici da DataFrame pandas.
    Usa colori multicolor hardcoded per massima semplicità.
    """

    # Schema - campi popolati automaticamente dal decoratore
    chart_type: Literal[
        "line", "bar", "scatter", "histogram", "pie", "box",
        "sunburst", "treemap"
    ] = Field(
        ...,
        description=(
            "Tipo di grafico da creare:\n"
            "- line: Serie temporali o trend (richiede x e y)\n"
            "- bar: Confronti categorici (richiede x e y)\n"
            "- scatter: Correlazione tra variabili (richiede x e y)\n"
            "- histogram: Distribuzione di singola variabile (richiede x)\n"
            "- pie: Relazioni parte-tutto (richiede x, y opzionale)\n"
            "- box: Distribuzione con outlier (richiede y, x opzionale per raggruppamento)\n"
            "- sunburst: Vista gerarchica circolare (richiede hierarchical_path e hierarchical_values)\n"
            "- treemap: Vista gerarchica rettangolare (richiede hierarchical_path e hierarchical_values)"
        )
    )

    title: str = Field(
        ...,
        description="Titolo chiaro e descrittivo che spiega cosa mostra il grafico"
    )

    subtitle: Optional[str] = Field(
        None,
        description="Sottotitolo opzionale per contesto aggiuntivo"
    )

    x_column: Optional[str] = Field(
        None,
        description="Nome colonna per asse X (richiesto per line, bar, scatter, histogram, pie)"
    )

    y_column: Optional[str] = Field(
        None,
        description="Nome colonna per asse Y (richiesto per line, bar, scatter, box)"
    )

    hierarchical_path: Optional[List[str]] = Field(
        None,
        description=(
            "Lista di nomi colonna che definiscono la gerarchia per sunburst/treemap. "
            "Esempio: ['regione', 'categoria', 'prodotto']. Richiesto per sunburst e treemap."
        )
    )

    hierarchical_values: Optional[str] = Field(
        None,
        description=(
            "Nome colonna per i valori nei grafici gerarchici. "
            "Determina le dimensioni dei segmenti. Richiesto per sunburst e treemap."
        )
    )

    top_n: Optional[int] = Field(
        None,
        description=(
            "Limita alle top N categorie per valore. "
            "Utile per dati categorici con molte categorie (>20) per mantenere i grafici leggibili."
        )
    )

    hover_data: Optional[List[str]] = Field(
        None,
        description=(
            "Lista di nomi colonna aggiuntivi da mostrare nel tooltip hover. "
            "Esempio: ['DESCRIZIONE_CAP', 'CAPITOLO'] mostrerà queste informazioni "
            "quando si passa il mouse sopra gli elementi del grafico (barre, punti, linee). "
            "Applicabile a: line, bar, scatter, histogram, box, sunburst, treemap."
        )
    )

    def __init__(self, dataframe: pd.DataFrame, output_dir: str = "output/charts"):
        """
        Inizializza il tool con il DataFrame da visualizzare.

        Args:
            dataframe: DataFrame pandas con i dati
            output_dir: Directory dove salvare i grafici (non usato, mantenuto per compatibilità)
        """
        self._df = dataframe.copy()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._chart_counter = 0

        # Genera tabella del DataFrame UNA SOLA VOLTA
        self._data_table_html = self._generate_master_data_table()

    async def execute(self) -> dict:
        """
        Esegue la creazione del grafico.

        Returns:
            dict con HTML del grafico creato e metadati
        """
        # Tutti gli attributi (chart_type, title, x_column, etc.) sono già popolati dal decoratore!

        # Valida colonne
        self._validate_columns()

        # Prepara DataFrame
        df = self._prepare_dataframe()

        if len(df) == 0:
            raise ValueError("No data available after filtering")

        # Crea il grafico
        fig = self._create_chart(df)

        # Genera HTML in memoria
        self._chart_counter += 1
        html_string = fig.to_html(
            include_plotlyjs='cdn',
            include_mathjax='cdn',
            full_html=False
        )

        # ============================================
        # DEBUG: Salva grafici su disco per ispezione
        # TODO: Rimuovere questo blocco in produzione
        # ============================================
        debug_dir = Path("output/charts/sales_example")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Salva HTML standalone per visualizzazione diretta
        debug_html_path = debug_dir / f"chart_{self._chart_counter}_{self.chart_type}.html"
        fig.write_html(str(debug_html_path), include_plotlyjs='cdn')

        # Salva anche come PNG se possibile (richiede kaleido)
        try:
            debug_png_path = debug_dir / f"chart_{self._chart_counter}_{self.chart_type}.png"
            fig.write_image(str(debug_png_path), width=1200, height=800)
        except Exception:
            # Kaleido non installato o errore, ignora silenziosamente
            pass
        # ============================================
        # FINE DEBUG
        # ============================================

        result_data = {
            "chart_html": html_string,
            "chart_type": self.chart_type,
            "data_points": len(df),
            "columns_used": {
                "x": self.x_column,
                "y": self.y_column
            },
            "message": f"Chart created successfully"
        }

        # Aggiungi tabella solo nel primo chart
        if self._chart_counter == 1 and self._data_table_html:
            result_data["data_table_html"] = self._data_table_html

        return result_data

    def _validate_columns(self) -> None:
        """Valida che le colonne specificate esistano nel DataFrame"""
        available_cols = set(self._df.columns)

        if self.x_column and self.x_column not in available_cols:
            raise ValueError(
                f"Column '{self.x_column}' not found. Available: {list(available_cols)}"
            )

        if self.y_column and self.y_column not in available_cols:
            raise ValueError(
                f"Column '{self.y_column}' not found. Available: {list(available_cols)}"
            )

        # Valida hierarchical columns
        if self.hierarchical_path:
            for col in self.hierarchical_path:
                if col not in available_cols:
                    raise ValueError(
                        f"Column '{col}' in hierarchical_path not found. Available: {list(available_cols)}"
                    )

        if self.hierarchical_values and self.hierarchical_values not in available_cols:
            raise ValueError(
                f"Column '{self.hierarchical_values}' not found. Available: {list(available_cols)}"
            )

        # Valida hover_data columns
        if self.hover_data:
            for col in self.hover_data:
                if col not in available_cols:
                    raise ValueError(
                        f"Column '{col}' in hover_data not found. Available: {list(available_cols)}"
                    )

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Prepara il DataFrame applicando filtri essenziali.
        """
        df = self._df.copy()

        # Rimuovi righe con NULL nelle colonne necessarie
        cols_to_check = []

        if self.x_column:
            cols_to_check.append(self.x_column)
        if self.y_column:
            cols_to_check.append(self.y_column)
        if self.hierarchical_path:
            cols_to_check.extend(self.hierarchical_path)
        if self.hierarchical_values:
            cols_to_check.append(self.hierarchical_values)

        # Rimuovi duplicati
        cols_to_check = list(set(cols_to_check))

        if cols_to_check:
            df = df.dropna(subset=cols_to_check)

        # Applica top_n filtering
        if self.top_n and self.x_column:
            x_col = self.x_column
            # Solo per dati categorici
            if df[x_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[x_col]):
                if self.y_column:
                    # Top N per somma del valore Y
                    top_categories = (
                        df.groupby(x_col)[self.y_column]
                        .sum()
                        .nlargest(self.top_n)
                        .index
                    )
                else:
                    # Top N per frequenza
                    top_categories = df[x_col].value_counts().head(self.top_n).index

                df = df[df[x_col].isin(top_categories)]

        return df

    def _create_chart(self, df: pd.DataFrame) -> Any:
        """Crea il grafico Plotly in base al tipo richiesto"""
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly not installed. Install with: pip install plotly")

        title = self._format_title(self.title, self.subtitle)

        # Dispatch ai metodi specifici
        if self.chart_type == "line":
            fig = self._create_line(df, px)
        elif self.chart_type == "bar":
            fig = self._create_bar(df, px)
        elif self.chart_type == "scatter":
            fig = self._create_scatter(df, px)
        elif self.chart_type == "histogram":
            fig = self._create_histogram(df, px)
        elif self.chart_type == "pie":
            fig = self._create_pie(df, px)
        elif self.chart_type == "box":
            fig = self._create_box(df, px)
        elif self.chart_type == "sunburst":
            fig = self._create_sunburst(df, px)
        elif self.chart_type == "treemap":
            fig = self._create_treemap(df, px)
        else:
            raise ValueError(f"Unsupported chart type: {self.chart_type}")

        # Applica layout standard
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            font=dict(size=12),
            showlegend=True,
            template="plotly_white",
            margin=dict(t=100, l=80, r=80, b=80),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )

        return fig

    def _format_title(self, title: str, subtitle: Optional[str]) -> str:
        """Formatta titolo con eventuale sottotitolo"""
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><sub>{subtitle}</sub>"
        return title_text

    def _create_line(self, df: pd.DataFrame, px) -> Any:
        """Crea line chart"""
        if not self.x_column or not self.y_column:
            raise ValueError("Line chart requires both x_column and y_column")

        fig = px.line(
            df,
            x=self.x_column,
            y=self.y_column,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        fig.update_traces(mode='lines+markers')
        return fig

    def _create_bar(self, df: pd.DataFrame, px) -> Any:
        """Crea bar chart"""
        if not self.x_column or not self.y_column:
            raise ValueError("Bar chart requires both x_column and y_column")

        fig = px.bar(
            df,
            x=self.x_column,
            y=self.y_column,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        return fig

    def _create_scatter(self, df: pd.DataFrame, px) -> Any:
        """Crea scatter plot"""
        if not self.x_column or not self.y_column:
            raise ValueError("Scatter plot requires both x_column and y_column")

        fig = px.scatter(
            df,
            x=self.x_column,
            y=self.y_column,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        return fig

    def _create_histogram(self, df: pd.DataFrame, px) -> Any:
        """Crea histogram"""
        if not self.x_column:
            raise ValueError("Histogram requires x_column")

        fig = px.histogram(
            df,
            x=self.x_column,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        return fig

    def _create_pie(self, df: pd.DataFrame, px) -> Any:
        """Crea pie chart"""
        if not self.x_column:
            raise ValueError("Pie chart requires x_column")

        if self.y_column:
            # Usa valori specificati
            fig = px.pie(
                df,
                names=self.x_column,
                values=self.y_column,
                color_discrete_sequence=MULTICOLOR_PALETTE
            )
        else:
            # Conta occorrenze
            value_counts = df[self.x_column].value_counts().reset_index()
            value_counts.columns = [self.x_column, 'count']
            fig = px.pie(
                value_counts,
                names=self.x_column,
                values='count',
                color_discrete_sequence=MULTICOLOR_PALETTE
            )

        return fig

    def _create_box(self, df: pd.DataFrame, px) -> Any:
        """Crea box plot"""
        if not self.y_column:
            raise ValueError("Box plot requires y_column")

        fig = px.box(
            df,
            x=self.x_column,
            y=self.y_column,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        return fig

    def _create_sunburst(self, df: pd.DataFrame, px) -> Any:
        """Crea sunburst chart"""
        if not self.hierarchical_path or not self.hierarchical_values:
            raise ValueError("Sunburst requires hierarchical_path and hierarchical_values")

        fig = px.sunburst(
            df,
            path=self.hierarchical_path,
            values=self.hierarchical_values,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )
        return fig

    def _create_treemap(self, df: pd.DataFrame, px) -> Any:
        """Crea treemap chart"""
        if not self.hierarchical_path or not self.hierarchical_values:
            raise ValueError("Treemap requires hierarchical_path and hierarchical_values")

        fig = px.treemap(
            df,
            path=self.hierarchical_path,
            values=self.hierarchical_values,
            hover_data=self.hover_data,
            color_discrete_sequence=MULTICOLOR_PALETTE
        )

        # Aggiungi percentuali nel hover
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percentParent}",
            hovertemplate="<b>%{label}</b><br>" +
                         "Value: %{value:,.0f}<br>" +
                         "% of Parent: %{percentParent}<br>" +
                         "% of Root: %{percentRoot}<br>" +
                         "<extra></extra>"
        )

        return fig

    def _generate_master_data_table(self) -> str:
        """
        Genera la tabella del DataFrame originale completo.
        Chiamato UNA SOLA VOLTA nel __init__.

        Returns:
            str: HTML della tabella
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return ""

        display_df = self._df.copy()
        max_rows = 500
        is_truncated = False

        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
            is_truncated = True

        # Formatta valori numerici
        formatted_values = []
        for col in display_df.columns:
            col_values = []
            for val in display_df[col]:
                if pd.isna(val):
                    col_values.append("N/A")
                elif isinstance(val, (int, float)):
                    if isinstance(val, float) and val != int(val):
                        col_values.append(f"{val:,.2f}")
                    else:
                        col_values.append(f"{int(val):,}")
                else:
                    col_values.append(str(val))
            formatted_values.append(col_values)

        # Colori tabella
        header_color = '#2c3e50'
        header_font_color = 'white'
        even_row_color = '#ecf0f1'
        odd_row_color = 'white'

        row_colors = [even_row_color if i % 2 == 0 else odd_row_color
                     for i in range(len(display_df))]

        # Crea tabella
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in display_df.columns],
                fill_color=header_color,
                font=dict(color=header_font_color, size=14, family="Arial"),
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
        table_title = f"<b>Complete Dataset Table</b>"
        if is_truncated:
            table_title += f"<br><sub>Showing first {max_rows} of {len(self._df):,} rows</sub>"
        else:
            table_title += f"<br><sub>Total rows: {len(self._df):,}</sub>"

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

        # Genera HTML
        table_html = fig.to_html(
            include_plotlyjs='cdn',
            include_mathjax='cdn',
            full_html=False
        )

        return table_html
