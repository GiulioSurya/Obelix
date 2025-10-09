from pydantic import Field
from typing import Dict, Any, List, Optional
from enum import Enum
import aiohttp
import json
import os
import time
import re

from src.tools.tool_schema import ToolSchema
from src.tools.tool_base import ToolBase
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus


class NotionParentType(str, Enum):
    """Tipi di parent supportati per le pagine Notion"""
    PAGE_ID = "page_id"
    DATA_SOURCE_ID = "data_source_id"


class NotionPageSchema(ToolSchema):
    """Schema per il tool di creazione pagine Notion con supporto blocchi avanzati"""

    tool_name = "notion_page"
    tool_description = """Create Notion pages with support for advanced blocks using extended Markdown"""

    title: str = Field(..., description="Notion's page title")
    content: Optional[str] = Field(..., description="""Page content must be provided in extended Markdown format, supporting the following blocks: Tables: |Col1|Col2| followed by |---|---| Callouts: :::info Text ::: Code blocks: <code>python code </code> Toggle blocks: ++ Title followed by content and ++ Quotes: > Quoted text Dividers: ---""")
    icon_emoji: Optional[str] = Field(None, description="Unicode emoji for the page icon (e.g., U+1F4C8, U+1F4CA, U+1F4DD). **CRITICAL** Do not use the :name: format.")


class NotionPageTool(ToolBase):
    """Tool per creare pagine Notion con contenuto markdown avanzato"""

    schema_class = NotionPageSchema

    def __init__(self, notion_token: Optional[str] = None, api_version: str = "2022-06-28"):
        """Inizializza il tool Notion"""
        if notion_token is None:
            notion_token = os.getenv("NOTION_TOKEN")

        if not notion_token:
            raise ValueError("Notion token is required. Set NOTION_TOKEN env var or pass notion_token parameter")

        self.notion_token = notion_token
        self.api_version = api_version
        self.base_url = "https://api.notion.com/v1"

        # Leggi configurazione parent dalle env vars
        self.parent_id = os.getenv("NOTION_PARENT_ID")
        parent_type_env = os.getenv("NOTION_PARENT_TYPE")

        if not self.parent_id:
            raise ValueError("NOTION_PARENT_ID env var is required")

        # Auto-detect parent_type se non specificato
        if parent_type_env:
            if parent_type_env not in ["page_id", "data_source_id"]:
                raise ValueError(f"Invalid NOTION_PARENT_TYPE: {parent_type_env}. Use 'page_id' or 'data_source_id'")
            self.parent_type = parent_type_env
        else:
            self.parent_type = self._auto_detect_parent_type(self.parent_id)

        print(f"[NotionPageTool] Configured with parent: {self.parent_type}={self.parent_id[:8]}...")

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Esegue la creazione di pagina Notion con parametri standardizzati"""
        start_time = time.time()

        try:
            validated_params = self.schema_class(**tool_call.arguments)
            result = await self._create_notion_page(validated_params)
            execution_time = time.time() - start_time

            if result.get("success", False):
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=result,
                    status=ToolStatus.SUCCESS,
                    execution_time=execution_time
                )
            else:
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=None,
                    status=ToolStatus.ERROR,
                    error=result.get("error", "Unknown error"),
                    execution_time=execution_time
                )

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

    def _auto_detect_parent_type(self, parent_id: str) -> str:
        """Auto-rileva il tipo di parent basandosi sul formato dell'ID"""
        parent_id = parent_id.strip()

        if parent_id.startswith("collection://"):
            return "data_source_id"

        clean_id = parent_id.replace("-", "")
        if len(clean_id) == 32 and all(c in "0123456789abcdefABCDEF" for c in clean_id):
            return "page_id"

        print(f"[NotionPageTool] Warning: Could not auto-detect parent type, assuming 'page_id'")
        return "page_id"

    async def _create_notion_page(self, params: NotionPageSchema) -> Dict[str, Any]:
        """Crea una pagina Notion con i parametri validati"""
        try:
            parent_type = NotionParentType(self.parent_type)
            parent_result = self._build_parent_object(parent_type, self.parent_id)
            if not parent_result["success"]:
                return {"success": False, "error": parent_result["error"]}

            # Costruisci properties con titolo
            properties = {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": params.title}
                        }
                    ]
                }
            }

            # Costruisci body richiesta
            body = {
                "parent": parent_result["parent"],
                "properties": properties
            }

            # Aggiungi icon se specificato
            if params.icon_emoji and params.icon_emoji.strip():
                body["icon"] = {
                    "type": "emoji",
                    "emoji": params.icon_emoji.strip()
                }

            # Converti content markdown in blocchi Notion (ora con supporto avanzato)
            if params.content and params.content.strip():
                body["children"] = self._convert_markdown_to_blocks(params.content)

            # Esegui richiesta API
            api_response = await self._make_notion_request("POST", "/pages", body)

            if api_response["success"]:
                data = api_response["data"]
                return {
                    "success": True,
                    "page_id": data.get("id"),
                    "url": data.get("url"),
                    "title": params.title,
                    "created_time": data.get("created_time"),
                    "has_content": bool(params.content and params.content.strip()),
                    "has_icon": bool(params.icon_emoji),
                    "parent_type": parent_type.value,
                    "parent_id": self.parent_id
                }
            else:
                error_msg = api_response["data"].get("message", f"API error: {api_response['status_code']}")
                return {"success": False, "error": f"Failed to create page: {error_msg}"}

        except Exception as e:
            return {"success": False, "error": f"Page creation error: {str(e)}"}

    def _build_parent_object(self, parent_type: NotionParentType, parent_id: str) -> Dict[str, Any]:
        """Costruisce parent object per richiesta API"""
        if parent_type == NotionParentType.PAGE_ID:
            return {
                "success": True,
                "parent": {"type": "page_id", "page_id": parent_id}
            }
        elif parent_type == NotionParentType.DATA_SOURCE_ID:
            return {
                "success": True,
                "parent": {"type": "data_source_id", "data_source_id": parent_id}
            }
        else:
            return {"success": False, "error": f"Invalid parent_type: {parent_type.value}"}

    def _convert_markdown_to_blocks(self, markdown: str) -> List[Dict[str, Any]]:
        """Converte markdown esteso in blocchi Notion - VERSIONE AVANZATA"""
        if not markdown or not markdown.strip():
            return []

        blocks = []
        lines = markdown.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Linee vuote - skip
            if not line:
                i += 1
                continue

            # 1. Table of Contents
            if line.strip() == '[TOC]':
                blocks.append({
                    "object": "block",
                    "type": "table_of_contents",
                    "table_of_contents": {}
                })
                i += 1
                continue

            # 2. Divider (---)
            if re.match(r'^-{3,}$', line.strip()):
                blocks.append({
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                })
                i += 1
                continue

            # 3. Callout (:::tipo testo :::)
            callout_match = re.match(r'^:::(\w+)\s*(.*?)\s*:::$', line.strip())
            if callout_match:
                callout_type, text = callout_match.groups()
                blocks.append(self._create_callout_block(text, callout_type))
                i += 1
                continue

            # 4. Code block (```language)
            if line.strip().startswith('```'):
                code_block, i = self._parse_code_block(lines, i)
                if code_block:
                    blocks.append(code_block)
                continue

            # 5. Toggle (++ Titolo)
            if line.strip().startswith('++ '):
                toggle_block, i = self._parse_toggle_block(lines, i)
                if toggle_block:
                    blocks.append(toggle_block)
                continue

            # 6. Table (|header|)
            if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
                table_block, i = self._parse_table_block(lines, i)
                if table_block:
                    blocks.append(table_block)
                continue

            # 7. Quote (> testo)
            if line.strip().startswith('> '):
                blocks.append(self._create_quote_block(line[2:].strip()))
                i += 1
                continue

            # Headers (H1, H2, H3) - mantiene metodi originali
            if line.strip().startswith('### '):
                blocks.append(self._create_heading_block(line.strip()[4:], "heading_3"))
                i += 1
                continue
            elif line.strip().startswith('## '):
                blocks.append(self._create_heading_block(line.strip()[3:], "heading_2"))
                i += 1
                continue
            elif line.strip().startswith('# '):
                blocks.append(self._create_heading_block(line.strip()[2:], "heading_1"))
                i += 1
                continue

            # Liste - mantiene metodi originali
            elif line.strip().startswith('- '):
                blocks.append(self._create_list_block(line.strip()[2:], "bulleted_list_item"))
                i += 1
                continue
            elif len(line.strip()) > 2 and line.strip()[0].isdigit() and line.strip()[1:3] == '. ':
                blocks.append(self._create_list_block(line.strip()[3:], "numbered_list_item"))
                i += 1
                continue

            # Paragrafo normale - mantiene metodo originale
            else:
                blocks.append(self._create_paragraph_block(line))
                i += 1

        return blocks

    # NUOVI METODI per blocchi avanzati
    def _create_callout_block(self, text: str, callout_type: str = "info") -> Dict[str, Any]:
        """Crea blocco callout con icona e colore"""
        color_map = {
            "info": "blue_background",
            "warning": "yellow_background",
            "error": "red_background",
            "success": "green_background",
            "note": "gray_background"
        }
        icon_map = {
            "info": "💡",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "note": "📝"
        }

        return {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "icon": {"type": "emoji", "emoji": icon_map.get(callout_type, "💡")},
                "color": color_map.get(callout_type, "blue_background")
            }
        }

    def _create_quote_block(self, text: str) -> Dict[str, Any]:
        """Crea blocco quote"""
        return {
            "object": "block",
            "type": "quote",
            "quote": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    def _parse_code_block(self, lines: List[str], start_idx: int) -> tuple[Optional[Dict[str, Any]], int]:
        """Parsa code block con syntax highlighting"""
        first_line = lines[start_idx].strip()
        language = first_line[3:].strip() or "plain text"

        code_lines = []
        i = start_idx + 1

        while i < len(lines) and lines[i].strip() != '```':
            code_lines.append(lines[i])
            i += 1

        if i >= len(lines):  # No closing ```
            return None, start_idx + 1

        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": '\n'.join(code_lines)}}],
                "language": language.lower()
            }
        }, i + 1

    def _parse_toggle_block(self, lines: List[str], start_idx: int) -> tuple[Optional[Dict[str, Any]], int]:
        """Parsa toggle block collassabile"""
        title = lines[start_idx][3:].strip()

        content_lines = []
        i = start_idx + 1

        while i < len(lines) and lines[i].strip() != '++':
            content_lines.append(lines[i])
            i += 1

        children = []
        if content_lines:
            content_text = '\n'.join(content_lines).strip()
            children = self._convert_markdown_to_blocks(content_text)

        return {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": title}}],
                "children": children
            }
        }, i + 1

    def _parse_table_block(self, lines: List[str], start_idx: int) -> tuple[Optional[Dict[str, Any]], int]:
        """Parsa tabelle Markdown in formato Notion"""
        header_line = lines[start_idx]
        header_cells = [cell.strip() for cell in header_line.strip().split('|')[1:-1]]

        if not header_cells:
            return None, start_idx + 1

        table_width = len(header_cells)
        i = start_idx + 1

        # Skip separator line se presente
        if i < len(lines) and re.match(r'^\|[\s\-\|]+\|$', lines[i]):
            i += 1

        # Raccogli righe dati
        table_rows = []

        while i < len(lines):
            line = lines[i].strip()
            if not (line.startswith('|') and line.endswith('|')):
                break

            row_cells = [cell.strip() for cell in line.split('|')[1:-1]]
            # Pad per matching width
            while len(row_cells) < table_width:
                row_cells.append("")

            table_rows.append({
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [[{"type": "text", "text": {"content": cell}}] for cell in row_cells[:table_width]]
                }
            })
            i += 1

        # Costruisci table con header
        all_rows = [
            {
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [[{"type": "text", "text": {"content": cell}}] for cell in header_cells]
                }
            }
        ]
        all_rows.extend(table_rows)

        return {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": table_width,
                "has_column_header": True,
                "has_row_header": False,
                "children": all_rows
            }
        }, i

    # METODI ORIGINALI - non modificati
    def _create_heading_block(self, text: str, heading_type: str) -> Dict[str, Any]:
        """Crea blocco heading"""
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }

    def _create_list_block(self, text: str, list_type: str) -> Dict[str, Any]:
        """Crea blocco lista"""
        return {
            "object": "block",
            "type": list_type,
            list_type: {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }

    def _create_paragraph_block(self, text: str) -> Dict[str, Any]:
        """Crea blocco paragrafo"""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }

    async def _make_notion_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Esegue richiesta HTTP a Notion API usando aiohttp"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.notion_token}",
            "Notion-Version": self.api_version,
            "Content-Type": "application/json"
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        return await self._process_response(response)
                elif method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        return await self._process_response(response)
                elif method.upper() == "PATCH":
                    async with session.patch(url, headers=headers, json=data) as response:
                        return await self._process_response(response)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

        except asyncio.TimeoutError:
            return {
                "status_code": 408,
                "data": {"error": "Request timeout"},
                "success": False
            }
        except aiohttp.ClientConnectionError:
            return {
                "status_code": 500,
                "data": {"error": "Connection error"},
                "success": False
            }
        except Exception as e:
            return {
                "status_code": 500,
                "data": {"error": f"Request failed: {str(e)}"},
                "success": False
            }

    async def _process_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Processa la risposta HTTP"""
        try:
            response_data = await response.json()
        except json.JSONDecodeError:
            response_data = {"error": "Invalid JSON response"}

        return {
            "status_code": response.status,
            "data": response_data,
            "success": 200 <= response.status < 300
        }


# Test del tool
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()


    async def test_notion_tool():
        try:
            tool = NotionPageTool()

            # Test content con blocchi avanzati
            test_content = """
            # Report Avanzato

            Questo è un report con blocchi avanzati.

            :::info Questo è un callout informativo :::

            :::warning Attenzione a questo avvertimento :::

            ## Codice Esempio

            ```python
            def hello_world():
                print("Hello, World!")
                return True
            ```

            ## Tabella Dati

            |Nome|Età|Città|
            |---|---|---|
            |Mario|30|Roma|
            |Luigi|25|Milano|
            |Peach|28|Torino|

            ++ Dettagli Avanzati
            Questo contenuto è nascosto in un toggle.

            - Lista dentro toggle
            - Altro elemento
            ++

            > "Questa è una citazione importante"

            ---

            [TOC]
            """

            print("Tool Notion avanzato configurato!")
            print("Supporta: callout, code, tabelle, toggle, quote, divider, TOC")

        except ValueError as e:
            print(f"Errore configurazione: {e}")


    asyncio.run(test_notion_tool())