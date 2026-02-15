"""
Utility per estrarre e cachare informazioni sullo schema del database.
Separato da QueryEnhancementAgent per evitare query lente ad ogni inizializzazione.

NOTA: Questo modulo usa connessioni Oracle SYNC, pensato per script offline
(generazione cache schema). NON usare nel flusso API - usare il pool async.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import oracledb

from sql.connections.db_connection import get_sync_connection, execute_query_sync


class DBSchemaExtractor:
    """
    Estrae informazioni su tabelle e colonne dal database e le salva in cache.

    Usa connessione Oracle sincrona (per script offline, non API).
    La connessione viene creata lazy alla prima query e riusata.
    Chiamare close() quando finito.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_distinct_values: int = 3,
                 full_distinct_columns: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            cache_dir: Directory dove salvare i file di cache (default: src/tools/utils/schema_cache)
            max_distinct_values: Numero massimo di valori distinti da estrarre (default 3)
            full_distinct_columns: Dizionario {table_name: [column_names]} per estrarre TUTTI i valori distinti
                                   Es: {"VISTA_BILANCIO_ENTRATA_AI": ["COD_STATO", "TIPO_BUDGET"]}
        """
        if cache_dir is None:
            # Default: directory cache/schema_cache nella root del modulo database
            self.cache_dir = Path(__file__).parent.parent / "cache" / "schema_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_distinct_values = max_distinct_values
        self.full_distinct_columns = full_distinct_columns or {}

        # Mantieni compatibilità con vecchio nome attributo
        self.max_sample_values = max_distinct_values

        # Connessione lazy (creata alla prima query)
        self._connection: Optional[oracledb.Connection] = None

    def _get_connection(self) -> oracledb.Connection:
        """Ritorna connessione Oracle, creandola se necessario."""
        if self._connection is None:
            self._connection = get_sync_connection()
        return self._connection

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> tuple:
        """Esegue query usando connessione lazy."""
        return execute_query_sync(self._get_connection(), query, params)

    def close(self) -> None:
        """Chiude la connessione Oracle se aperta."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def extract_table_info(self, name_filter=None, table_list=None, include_tables=True, include_views=True):
        """
        Recupera informazioni su tabelle e viste disponibili nel database.
        Stesso codice di QueryEnhancementAgent.retrive_table_info()
        """
        if table_list:
            queries = []

            table_query = """
                SELECT
                    t.table_name AS object_name,
                    'TABLE' AS object_type,
                    c.comments AS description
                FROM user_tables t
                LEFT JOIN user_tab_comments c
                    ON t.table_name = c.table_name
                    AND c.table_type = 'TABLE'
            """
            queries.append(table_query)

            view_query = """
                SELECT
                    v.view_name AS object_name,
                    'VIEW' AS object_type,
                    c.comments AS description
                FROM user_views v
                LEFT JOIN user_tab_comments c
                    ON v.view_name = c.table_name
                    AND c.table_type = 'VIEW'
            """
            queries.append(view_query)

            full_query = "\nUNION ALL\n".join(queries)
            placeholders = ', '.join([f":table_{i}" for i in range(len(table_list))])
            full_query = f"SELECT * FROM ({full_query}) WHERE object_name IN ({placeholders})"
            full_query += "\nORDER BY object_type, object_name"

            try:
                params = {f'table_{i}': table_name.upper() for i, table_name in enumerate(table_list)}
                results, _ = self._execute_query(full_query, params)

                if not results:
                    return "AVAILABLE_TABLES: none\n"

                formatted_info = "AVAILABLE_TABLES:\n"
                for row in results:
                    desc = row[2] if row[2] else "N/A"
                    formatted_info += f"- table_name:{row[0]} type:{row[1]} description:{desc}\n"

                return formatted_info

            except Exception as e:
                return f"Errore nel recupero delle informazioni sulle tabelle: {str(e)}"

        queries = []

        if include_tables:
            queries.append("""
                SELECT
                    t.table_name AS object_name,
                    'TABLE' AS object_type,
                    c.comments AS description
                FROM user_tables t
                LEFT JOIN user_tab_comments c
                    ON t.table_name = c.table_name
                    AND c.table_type = 'TABLE'
            """)

        if include_views:
            queries.append("""
                SELECT
                    v.view_name AS object_name,
                    'VIEW' AS object_type,
                    c.comments AS description
                FROM user_views v
                LEFT JOIN user_tab_comments c
                    ON v.view_name = c.table_name
                    AND c.table_type = 'VIEW'
            """)

        if not queries:
            return "## TABELLE E VISTE DISPONIBILI:\n\nNessun tipo di oggetto selezionato.\n"

        full_query = "\nUNION ALL\n".join(queries)
        if name_filter:
            full_query = f"SELECT * FROM ({full_query}) WHERE object_name LIKE :name_filter"
        full_query += "\nORDER BY object_type, object_name"

        try:
            if name_filter:
                results, _ = self._execute_query(full_query, {'name_filter': name_filter.upper()})
            else:
                results, _ = self._execute_query(full_query)

            if not results:
                return "AVAILABLE_TABLES: none\n"

            formatted_info = "AVAILABLE_TABLES:\n"
            for row in results:
                desc = row[2] if row[2] else "N/A"
                formatted_info += f"- table_name:{row[0]} type:{row[1]} description:{desc}\n"

            return formatted_info

        except Exception as e:
            return f"Errore nel recupero delle informazioni sulle tabelle: {str(e)}"

    def extract_columns_info(self, name_filter=None, table_list=None, include_tables=True, include_views=True):
        """
        Recupera informazioni dettagliate sulle colonne di tabelle/viste.
        Versione modificata con supporto per tutte le colonne VARCHAR2 e NUMBER.
        """
        try:
            if table_list:
                queries = [
                    "SELECT table_name AS object_name FROM user_tables",
                    "SELECT view_name AS object_name FROM user_views"
                ]
                tables_query = "\nUNION ALL\n".join(queries)
                placeholders = ', '.join([f":table_{i}" for i in range(len(table_list))])
                tables_query = f"SELECT * FROM ({tables_query}) WHERE object_name IN ({placeholders}) ORDER BY object_name"
                params = {f'table_{i}': table_name.upper() for i, table_name in enumerate(table_list)}
                tables_results, _ = self._execute_query(tables_query, params)
            else:
                queries = []
                if include_tables:
                    queries.append("SELECT table_name AS object_name FROM user_tables")
                if include_views:
                    queries.append("SELECT view_name AS object_name FROM user_views")

                if not queries:
                    return "## COLONNE DELLE TABELLE:\n\nNessun tipo di oggetto selezionato.\n"

                tables_query = "\nUNION ALL\n".join(queries)
                if name_filter:
                    tables_query = f"SELECT * FROM ({tables_query}) WHERE object_name LIKE :name_filter ORDER BY object_name"
                    tables_results, _ = self._execute_query(tables_query, {'name_filter': name_filter.upper()})
                else:
                    tables_query += "\nORDER BY object_name"
                    tables_results, _ = self._execute_query(tables_query)

            formatted_info = "DATABASE_SCHEMA:\n"

            for table_row in tables_results:
                table_name = table_row[0]
                formatted_info += f"\n[TABLE:{table_name}]\n"

                columns_query = """
                    SELECT c.column_name,c.data_type,c.data_length,c.data_precision,c.data_scale,cc.comments
                    FROM user_tab_columns c
                    LEFT JOIN user_col_comments cc ON c.table_name=cc.table_name AND c.column_name=cc.column_name
                    WHERE c.table_name=:table_name ORDER BY c.column_id
                """
                columns_results, _ = self._execute_query(columns_query, {'table_name': table_name})

                pk_query = """
                    SELECT cc.column_name FROM user_constraints c
                    JOIN user_cons_columns cc ON c.constraint_name=cc.constraint_name
                    WHERE c.table_name=:table_name AND c.constraint_type='P'
                """
                pk_results, _ = self._execute_query(pk_query, {'table_name': table_name})
                pk_columns = {row[0] for row in pk_results}

                for col_row in columns_results:
                    col_name = col_row[0]
                    dtype = col_row[1]
                    dlen = col_row[2]
                    dprec = col_row[3]
                    dscale = col_row[4]
                    desc = col_row[5] if col_row[5] else ""

                    if dprec is not None:
                        tdisplay = f"{dtype}({dprec},{dscale})" if dscale and dscale > 0 else f"{dtype}({dprec})"
                    elif dlen and dtype in ('VARCHAR2','CHAR','NVARCHAR2','NCHAR'):
                        tdisplay = f"{dtype}({dlen})"
                    else:
                        tdisplay = dtype

                    col_info = f"column_name:{col_name} type:{tdisplay}"

                    if col_name in pk_columns:
                        col_info += " is_primary_key:yes"

                    desc_display = desc if desc else "N/A"
                    col_info += f" description:{desc_display}"

                    # Analizza tutte le colonne stringa e numeriche per valori campione
                    is_string = dtype in ('VARCHAR2','CHAR','NVARCHAR2','NCHAR')
                    is_number = dtype in ('NUMBER',)

                    if is_string or is_number:
                        try:
                            # Mostra i top N valori più frequenti (configurabile)
                            top_query = f"SELECT * FROM (SELECT {col_name},COUNT(*) as f FROM {table_name} WHERE {col_name} IS NOT NULL GROUP BY {col_name} ORDER BY f DESC) WHERE ROWNUM<={self.max_sample_values}"
                            top_results, _ = self._execute_query(top_query)

                            if top_results:
                                tops = ','.join([f"{row[0]}({row[1]}occ)" for row in top_results])
                                col_info += f" most_frequent_values:[{tops}]"
                        except:
                            pass

                    formatted_info += f"{col_info}\n"

            return formatted_info

        except Exception as e:
            return f"Errore nel recupero delle informazioni sulle colonne: {str(e)}"

    def extract_schema_json(self, name_filter=None, table_list=None, include_tables=True, include_views=True) -> Dict[str, Any]:
        """
        Estrae schema in formato JSON strutturato (ottimizzato per LLM).

        Returns:
            Dict con struttura:
            {
                "tables": [
                    {
                        "table_name": "...",
                        "type": "TABLE/VIEW",
                        "description": "...",
                        "columns": [
                            {
                                "column_name": "...",
                                "type": "VARCHAR2(100)",
                                "description": "...",
                                "sample_values": ["val1(100occ)", "val2(50occ)", ...],
                                "is_primary_key": True,  # Presente solo se true
                                "other_distinct_values": 10  # Valori distinti NON mostrati in sample_values (omesso se 0)
                            }
                        ]
                    }
                ]
            }
        """
        # Step 1: Ottieni lista tabelle
        if table_list:
            queries = [
                "SELECT table_name AS object_name, 'TABLE' AS object_type FROM user_tables",
                "SELECT view_name AS object_name, 'VIEW' AS object_type FROM user_views"
            ]
            tables_query = "\nUNION ALL\n".join(queries)
            placeholders = ', '.join([f":table_{i}" for i in range(len(table_list))])
            tables_query = f"SELECT * FROM ({tables_query}) WHERE object_name IN ({placeholders}) ORDER BY object_name"
            params = {f'table_{i}': table_name.upper() for i, table_name in enumerate(table_list)}
            tables_results, _ = self._execute_query(tables_query, params)
        else:
            queries = []
            if include_tables:
                queries.append("SELECT table_name AS object_name, 'TABLE' AS object_type FROM user_tables")
            if include_views:
                queries.append("SELECT view_name AS object_name, 'VIEW' AS object_type FROM user_views")

            if not queries:
                return {"tables": []}

            tables_query = "\nUNION ALL\n".join(queries)
            if name_filter:
                tables_query = f"SELECT * FROM ({tables_query}) WHERE object_name LIKE :name_filter ORDER BY object_name"
                tables_results, _ = self._execute_query(tables_query, {'name_filter': name_filter.upper()})
            else:
                tables_query += "\nORDER BY object_name"
                tables_results, _ = self._execute_query(tables_query)

        # Step 2: Per ogni tabella, estrai colonne
        result = {"tables": []}

        for table_row in tables_results:
            table_name = table_row[0]
            table_type = table_row[1]

            print(f"  Elaborando {table_name}...")

            # Ottieni descrizione tabella
            desc_query = "SELECT comments FROM user_tab_comments WHERE table_name = :table_name"
            desc_results, _ = self._execute_query(desc_query, {'table_name': table_name})
            table_desc = desc_results[0][0] if desc_results and desc_results[0][0] else "N/A"

            # Ottieni colonne
            columns_query = """
                SELECT c.column_name, c.data_type, c.data_length, c.data_precision, c.data_scale, cc.comments
                FROM user_tab_columns c
                LEFT JOIN user_col_comments cc ON c.table_name=cc.table_name AND c.column_name=cc.column_name
                WHERE c.table_name=:table_name ORDER BY c.column_id
            """
            columns_results, _ = self._execute_query(columns_query, {'table_name': table_name})

            # Ottieni primary key
            pk_query = """
                SELECT cc.column_name FROM user_constraints c
                JOIN user_cons_columns cc ON c.constraint_name=cc.constraint_name
                WHERE c.table_name=:table_name AND c.constraint_type='P'
            """
            pk_results, _ = self._execute_query(pk_query, {'table_name': table_name})
            pk_columns = {row[0] for row in pk_results}

            columns_list = []

            for col_row in columns_results:
                col_name = col_row[0]
                dtype = col_row[1]
                dlen = col_row[2]
                dprec = col_row[3]
                dscale = col_row[4]
                desc = col_row[5] if col_row[5] else "N/A"

                # Type display
                if dprec is not None:
                    type_display = f"{dtype}({dprec},{dscale})" if dscale and dscale > 0 else f"{dtype}({dprec})"
                elif dlen and dtype in ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR'):
                    type_display = f"{dtype}({dlen})"
                else:
                    type_display = dtype

                # Sample values per colonne stringa e numeriche
                sample_values = []
                other_distinct_values = None
                is_string = dtype in ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR')
                is_number = dtype in ('NUMBER',)

                if is_string:
                    try:
                        # Determina se estrarre TUTTI i valori distinti per questa colonna
                        # Verifica se tabella e colonna sono in full_distinct_columns
                        table_cols = self.full_distinct_columns.get(table_name.upper(), [])
                        extract_all = col_name in [c.upper() for c in table_cols]

                        if extract_all:
                            # Estrai TUTTI i valori distinti (ordinati per frequenza)
                            all_query = f"SELECT {col_name},COUNT(*) as f FROM {table_name} WHERE {col_name} IS NOT NULL GROUP BY {col_name} ORDER BY f DESC"
                            all_results, _ = self._execute_query(all_query)

                            if all_results:
                                sample_values = [str(row[0]) for row in all_results]
                                # Nessun other_distinct_values perché abbiamo estratto tutto
                                other_distinct_values = 0
                        else:
                            # Estrai solo top N valori più frequenti (default)
                            top_query = f"SELECT * FROM (SELECT {col_name},COUNT(*) as f FROM {table_name} WHERE {col_name} IS NOT NULL GROUP BY {col_name} ORDER BY f DESC) WHERE ROWNUM<={self.max_distinct_values}"
                            top_results, _ = self._execute_query(top_query)

                            if top_results:
                                sample_values = [str(row[0]) for row in top_results]

                            # Query per contare valori distinti totali
                            count_query = f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name} WHERE {col_name} IS NOT NULL"
                            count_results, _ = self._execute_query(count_query)

                            if count_results and count_results[0][0] is not None:
                                total_distinct = int(count_results[0][0])
                                # Calcola "altri" valori non mostrati nei sample
                                other_distinct_values = total_distinct - len(sample_values)
                    except:
                        pass

                # Costruisci dizionario colonna
                col_dict = {
                    "column_name": col_name,
                    "type": type_display,
                    "description": desc,
                    "sample_values": sample_values
                }

                # Aggiungi is_primary_key SOLO se è true
                if col_name in pk_columns:
                    col_dict["is_primary_key"] = True

                # Aggiungi other_distinct_values SOLO se > 0
                if other_distinct_values is not None and other_distinct_values > 0:
                    col_dict["other_distinct_values"] = other_distinct_values

                columns_list.append(col_dict)
                print(f"    ✓ {col_name} ({type_display})")

            result["tables"].append({
                "table_name": table_name,
                "type": table_type,
                "description": table_desc,
                "columns": columns_list
            })

        return result

    def extract_and_save(self, name_filter=None, table_list=None, include_tables=True, include_views=True,
                         cache_name: Optional[str] = None, format='json'):
        """
        Estrae le informazioni dal DB e le salva in un file di cache.

        Args:
            name_filter: Filtro LIKE per nomi tabelle/viste
            table_list: Lista esplicita di tabelle/viste
            include_tables: Include tabelle
            include_views: Include viste
            cache_name: Nome del file di cache (default: auto-generato da table_list)
            format: 'json' o 'text' (default: 'json')

        Returns:
            Path del file salvato
        """
        if format == 'json':
            # Estrazione JSON
            print("Estrazione schema in formato JSON...")
            schema_data = self.extract_schema_json(name_filter, table_list, include_tables, include_views)

            # Genera nome file cache
            if cache_name is None:
                if table_list:
                    cache_name = "_".join(table_list) + "_schema.json"
                elif name_filter:
                    cache_name = f"filter_{name_filter}_schema.json"
                else:
                    cache_name = "full_schema.json"

            cache_file = self.cache_dir / cache_name

            # Salva JSON
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(schema_data, f, indent=2, ensure_ascii=False)

        else:
            # Estrazione testo (vecchio formato)
            print("Estrazione informazioni tabelle...")
            table_info = self.extract_table_info(name_filter, table_list, include_tables, include_views)

            print("Estrazione informazioni colonne (può richiedere tempo)...")
            columns_info = self.extract_columns_info(name_filter, table_list, include_tables, include_views)

            # Genera nome file cache
            if cache_name is None:
                if table_list:
                    cache_name = "_".join(table_list) + "_schema.txt"
                elif name_filter:
                    cache_name = f"filter_{name_filter}_schema.txt"
                else:
                    cache_name = "full_schema.txt"

            cache_file = self.cache_dir / cache_name

            # Salva in formato testo semplice
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write("# Schema Database Cache\n")
                f.write(f"# Max sample values: {self.max_sample_values}\n")
                f.write(f"# Generated with DBSchemaExtractor\n\n")
                f.write(table_info)
                f.write("\n")
                f.write(columns_info)

        print(f"Schema salvato in: {cache_file}")
        return cache_file

    @staticmethod
    def load_from_cache(cache_file: Path) -> str:
        """
        Carica informazioni da un file di cache (JSON, DDL o testo).

        Returns:
            - Se JSON: stringa JSON compatta (senza spazi, separators=(',',':'))
            - Se DDL (.sql): restituisce il contenuto DDL direttamente
            - Se testo: formato testo legacy (non più usato)
        """
        cache_path = Path(cache_file)

        # Determina formato da estensione
        if cache_path.suffix == '.json':
            # Carica JSON e restituisci versione compatta
            with open(cache_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)

            # Restituisci JSON compatto (massimo risparmio token)
            return json.dumps(schema_data, separators=(',', ':'), ensure_ascii=False)

        else:
            # Formato testo (vecchio, retro-compatibilità) o DDL
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Controlla se è formato DDL (generato da DDLSchemaGenerator)
            if cache_path.suffix == '.sql' or 'CREATE TABLE' in content[:1000]:
                # Formato DDL: restituisci direttamente
                return content

            # Separa table_info e columns_info (formato legacy)
            parts = content.split("DATABASE_SCHEMA:")

            if len(parts) < 2:
                raise ValueError(f"File cache malformato: {cache_file}")

            # Estrai table_information (tutto prima di DATABASE_SCHEMA)
            table_info_raw = parts[0]
            lines = table_info_raw.split('\n')
            table_info_lines = [l for l in lines if l.startswith('AVAILABLE_TABLES:') or l.startswith('- table_name:')]
            table_information = '\n'.join(table_info_lines)

            # Estrai columns_information
            columns_information = "DATABASE_SCHEMA:" + parts[1]

            # Formato legacy: concatena tutto
            return table_information + "\n" + columns_information


if __name__ == "__main__":
    """
    Script per rigenerare la cache dello schema.
    Esegui: python -m src.tools.utils.db_schema_extractor
    """
    from dotenv import load_dotenv

    load_dotenv()

    # Crea extractor con configurazione personalizzata
    # max_distinct_values: numero di valori da estrarre per colonne generiche (default 3)
    # full_distinct_columns: dizionario per specificare colonne da estrarre completamente
    extractor = DBSchemaExtractor(
        max_distinct_values=3,
        full_distinct_columns={
            'VISTA_BILANCIO_ENTRATA_AI': ["DESCRIZIONE_CAP", "DESCRIZIONE_CAP_ABB", "DES_TITOLO", "DES_TIPOLOGIA", "DES_COD_LIVELLO_1",
                                          "DES_COD_LIVELLO_2", "DES_COD_LIVELLO_3", "DES_COD_LIVELLO_4", "DES_COD_LIVELLO_5",
                                          "DES_VINCOLO", "UNITA_ORGANIZZATIVA", "RESPONSABILE_UO", "DES_PROGRAMMA", "DES_PROGETTO", "RESPONSABILE",
                                          "SE_UNA_TANTUM", "DES_CENTRO_COSTO", "SE_RILEV_IVA", "SE_FUNZ_DELEG", "SE_CONTRIB_COMU", "SE_RISORSA_SIGNIF",
                                          "FLESSIBILITA", "DES_FLESSIBILITA", "DES_FATTORE", "DES_CENTRO", "DES_CGE", "DES_OPERA_LIGHT", "DES_FINANZIAMENTO_LIGHT",
                                          "OTTICA", "SETTORE"],
            'VISTA_BILANCIO_SPESA_AI': ["DESCRIZIONE_CAP", "DESCRIZIONE_CAP_ABB", "DES_MISSIONE", "DES_PROGRAMMA_ARM", "DES_COD_LIVELLO_1",
                                        "DES_COD_LIVELLO_2", "DES_COD_LIVELLO_3", "DES_COD_LIVELLO_4", "DES_COD_LIVELLO_5",
                                        "DES_VINCOLO", "UNITA_ORGANIZZATIVA", "RESPONSABILE_UO", "DES_PROGRAMA", "DES_PROGETTO", "RESPONSABILE",
                                        "DES_CENTRO_COSTO", "FLESSIBILITA", "DES_FLESSIBILITA", "DES_FATTORE", "DES_CENTRO", "DES_CGU", "DES_OPERA_LIGHT",
                                        "DES_FINANZIAMENTO_LIGHT", "OTTICA","SETTORE"]
        }
    )

    # Estrai schema per le tabelle usate da QueryEnhancementAgent in formato JSON
    cache_file = extractor.extract_and_save(
        table_list=['vista_bilancio_entrata_ai', 'vista_bilancio_spesa_ai'],
        cache_name='bilancio_schema.json',
        format='json'
    )

    # Genera Literal Python dalla cache
    from obelix.database.schema.literal_generator import generate_literals_from_cache
    output_path = Path(__file__).parent / "generated" / "schema_literals.py"
    generate_literals_from_cache(cache_file, output_path)
