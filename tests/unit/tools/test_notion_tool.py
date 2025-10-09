"""
Unit tests for NotionPageTool.

Tests verify:
- Schema generation and validation
- Tool initialization with environment variables
- Page creation with mocked Notion API
- Markdown to Notion blocks conversion
- Error handling for API failures
- Parent type auto-detection
- Complex content parsing (tables, code blocks, callouts, etc.)

Note: All API calls are mocked - no real Notion API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

from src.tools.tool.notion_tool import (
    NotionPageTool,
    NotionPageSchema,
    NotionParentType
)
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus


class TestNotionToolSchema:
    """Test NotionPageSchema validation and metadata."""

    def test_notion_schema_generation(self):
        """Test that Notion tool generates correct MCP schema."""
        schema = NotionPageTool.create_schema()

        assert schema.name == "notion_page"
        assert "notion" in schema.description.lower()
        assert schema.inputSchema is not None

    def test_notion_schema_has_required_fields(self):
        """Test that schema has required fields."""
        schema = NotionPageTool.create_schema()
        input_props = schema.inputSchema["properties"]

        assert "title" in input_props
        assert "content" in input_props

    def test_notion_schema_validation_with_valid_data(self):
        """Test schema validates correct data."""
        valid_data = {
            "title": "Test Page",
            "content": "# Hello\n\nThis is content",
            "icon_emoji": "📄"
        }
        schema_instance = NotionPageSchema(**valid_data)

        assert schema_instance.title == "Test Page"
        assert schema_instance.content == "# Hello\n\nThis is content"
        assert schema_instance.icon_emoji == "📄"

    def test_notion_schema_title_required(self):
        """Test that title is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NotionPageSchema(content="Some content")  # Missing title

    def test_notion_schema_optional_fields(self):
        """Test that content and icon_emoji are optional."""
        schema_instance = NotionPageSchema(title="Just Title", content=None)

        assert schema_instance.title == "Just Title"
        assert schema_instance.content is None


class TestNotionToolInitialization:
    """Test NotionPageTool initialization and configuration."""

    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token_123", "NOTION_PARENT_ID": "page-id-123"})
    def test_initialization_with_env_vars(self):
        """Test tool initialization with environment variables."""
        tool = NotionPageTool()

        assert tool.notion_token == "test_token_123"
        assert tool.parent_id == "page-id-123"
        assert tool.parent_type == "page_id"  # Auto-detected
        assert tool.base_url == "https://api.notion.com/v1"

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_without_token_raises_error(self):
        """Test that missing NOTION_TOKEN raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            NotionPageTool()

        assert "token" in str(exc_info.value).lower()

    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token"}, clear=True)
    def test_initialization_without_parent_id_raises_error(self):
        """Test that missing NOTION_PARENT_ID raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            NotionPageTool()

        assert "parent_id" in str(exc_info.value).lower()

    def test_initialization_with_explicit_token(self):
        """Test initialization with explicit notion_token parameter."""
        with patch.dict(os.environ, {"NOTION_PARENT_ID": "page-123"}):
            tool = NotionPageTool(notion_token="explicit_token")

            assert tool.notion_token == "explicit_token"

    @patch.dict(os.environ, {
        "NOTION_TOKEN": "token",
        "NOTION_PARENT_ID": "page-123",
        "NOTION_PARENT_TYPE": "page_id"
    })
    def test_initialization_with_explicit_parent_type(self):
        """Test initialization with explicit NOTION_PARENT_TYPE."""
        tool = NotionPageTool()

        assert tool.parent_type == "page_id"

    @patch.dict(os.environ, {
        "NOTION_TOKEN": "token",
        "NOTION_PARENT_ID": "collection://abc123",
        "NOTION_PARENT_TYPE": "data_source_id"
    })
    def test_initialization_with_data_source_parent(self):
        """Test initialization with data_source_id parent type."""
        tool = NotionPageTool()

        assert tool.parent_type == "data_source_id"

    @patch.dict(os.environ, {
        "NOTION_TOKEN": "token",
        "NOTION_PARENT_ID": "page-123",
        "NOTION_PARENT_TYPE": "invalid_type"
    })
    def test_initialization_with_invalid_parent_type_raises_error(self):
        """Test that invalid parent type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            NotionPageTool()

        assert "invalid" in str(exc_info.value).lower()


class TestNotionToolParentTypeDetection:
    """Test auto-detection of parent type."""

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "abc123"})
    def test_auto_detect_page_id_format(self):
        """Test auto-detection recognizes page_id format."""
        tool = NotionPageTool()

        # 32-char hex string without dashes
        detected = tool._auto_detect_parent_type("a" * 32)
        assert detected == "page_id"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "abc123"})
    def test_auto_detect_page_id_with_dashes(self):
        """Test auto-detection handles page_id with dashes."""
        tool = NotionPageTool()

        page_id = "12345678-1234-1234-1234-123456789012"
        detected = tool._auto_detect_parent_type(page_id)
        assert detected == "page_id"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "abc123"})
    def test_auto_detect_data_source_id_format(self):
        """Test auto-detection recognizes data_source_id format."""
        tool = NotionPageTool()

        detected = tool._auto_detect_parent_type("collection://abc123")
        assert detected == "data_source_id"


class TestNotionToolPageCreation:
    """Test page creation with mocked Notion API."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_PARENT_ID": "parent-123"})
    async def test_create_page_success(self, mocker):
        """Test successful page creation."""
        # Mock the API request
        mock_response = {
            "success": True,
            "status_code": 200,
            "data": {
                "id": "page-id-456",
                "url": "https://notion.so/page-id-456",
                "created_time": "2024-01-01T00:00:00.000Z"
            }
        }

        tool = NotionPageTool()
        mocker.patch.object(tool, '_make_notion_request', return_value=mock_response)

        # Create tool call
        call = ToolCall(
            id="call_notion_1",
            name="notion_page",
            arguments={
                "title": "Test Page",
                "content": "# Hello World",
                "icon_emoji": "📄"
            }
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result["success"] is True
        assert result.result["page_id"] == "page-id-456"
        assert result.result["url"] == "https://notion.so/page-id-456"
        assert result.result["title"] == "Test Page"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_PARENT_ID": "parent-123"})
    async def test_create_page_without_content(self, mocker):
        """Test creating page with only title."""
        mock_response = {
            "success": True,
            "status_code": 200,
            "data": {
                "id": "page-id-789",
                "url": "https://notion.so/page-id-789",
                "created_time": "2024-01-01T00:00:00.000Z"
            }
        }

        tool = NotionPageTool()
        mocker.patch.object(tool, '_make_notion_request', return_value=mock_response)

        call = ToolCall(
            id="call_notion_2",
            name="notion_page",
            arguments={"title": "Title Only", "content": None}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.SUCCESS
        assert result.result["has_content"] is False

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_PARENT_ID": "parent-123"})
    async def test_create_page_api_error(self, mocker):
        """Test handling of API error response."""
        mock_response = {
            "success": False,
            "status_code": 400,
            "data": {"message": "Invalid request"}
        }

        tool = NotionPageTool()
        mocker.patch.object(tool, '_make_notion_request', return_value=mock_response)

        call = ToolCall(
            id="call_notion_3",
            name="notion_page",
            arguments={"title": "Test", "content": "Content"}
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert "Invalid request" in result.error

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_PARENT_ID": "parent-123"})
    async def test_create_page_missing_required_field(self, mocker):
        """Test error when missing required field (title)."""
        tool = NotionPageTool()

        call = ToolCall(
            id="call_notion_4",
            name="notion_page",
            arguments={"content": "Content only"}  # Missing title
        )

        result = await tool.execute(call)

        assert result.status == ToolStatus.ERROR
        assert result.error is not None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_PARENT_ID": "parent-123"})
    async def test_create_page_tracks_execution_time(self, mocker):
        """Test that execution time is tracked."""
        mock_response = {
            "success": True,
            "status_code": 200,
            "data": {
                "id": "page-id-time",
                "url": "https://notion.so/page-id-time",
                "created_time": "2024-01-01T00:00:00.000Z"
            }
        }

        tool = NotionPageTool()
        mocker.patch.object(tool, '_make_notion_request', return_value=mock_response)

        call = ToolCall(
            id="call_time",
            name="notion_page",
            arguments={"title": "Time Test", "content": None}
        )

        result = await tool.execute(call)

        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestNotionToolMarkdownConversion:
    """Test markdown to Notion blocks conversion."""

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_heading_blocks(self):
        """Test conversion of markdown headings."""
        tool = NotionPageTool()

        markdown = "# H1\n## H2\n### H3"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 3
        assert blocks[0]["type"] == "heading_1"
        assert blocks[1]["type"] == "heading_2"
        assert blocks[2]["type"] == "heading_3"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_paragraph_block(self):
        """Test conversion of plain paragraph."""
        tool = NotionPageTool()

        markdown = "This is a paragraph"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "This is a paragraph"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_bulleted_list(self):
        """Test conversion of bulleted list."""
        tool = NotionPageTool()

        markdown = "- Item 1\n- Item 2\n- Item 3"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 3
        assert all(b["type"] == "bulleted_list_item" for b in blocks)

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_numbered_list(self):
        """Test conversion of numbered list."""
        tool = NotionPageTool()

        markdown = "1. First\n2. Second\n3. Third"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 3
        assert all(b["type"] == "numbered_list_item" for b in blocks)

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_divider(self):
        """Test conversion of divider (---)."""
        tool = NotionPageTool()

        markdown = "---"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "divider"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_quote_block(self):
        """Test conversion of quote."""
        tool = NotionPageTool()

        markdown = "> This is a quote"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "quote"
        assert "This is a quote" in blocks[0]["quote"]["rich_text"][0]["text"]["content"]

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_callout_block(self):
        """Test conversion of callout."""
        tool = NotionPageTool()

        markdown = ":::info This is important :::"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "callout"
        assert blocks[0]["callout"]["icon"]["emoji"] == "💡"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_code_block(self):
        """Test conversion of code block with language."""
        tool = NotionPageTool()

        markdown = "```python\ndef hello():\n    print('hi')\n```"
        blocks = tool._convert_markdown_to_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "python"
        assert "def hello()" in blocks[0]["code"]["rich_text"][0]["text"]["content"]

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_empty_content(self):
        """Test conversion of empty/None content."""
        tool = NotionPageTool()

        blocks = tool._convert_markdown_to_blocks(None)
        assert blocks == []

        blocks = tool._convert_markdown_to_blocks("")
        assert blocks == []

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_convert_mixed_content(self):
        """Test conversion of mixed markdown elements."""
        tool = NotionPageTool()

        markdown = """# Title

        Paragraph text

        - List item

        > Quote"""

        blocks = tool._convert_markdown_to_blocks(markdown)

        # Should have heading, paragraph, list item, quote
        assert len(blocks) >= 4
        types = [b["type"] for b in blocks]
        assert "heading_1" in types
        assert "paragraph" in types
        assert "bulleted_list_item" in types
        assert "quote" in types


class TestNotionToolResultStructure:
    """Test ToolResult structure and metadata."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    async def test_result_structure_on_success(self, mocker):
        """Test complete ToolResult structure on success."""
        mock_response = {
            "success": True,
            "status_code": 200,
            "data": {
                "id": "page-struct-test",
                "url": "https://notion.so/page-struct-test",
                "created_time": "2024-01-01T00:00:00.000Z"
            }
        }

        tool = NotionPageTool()
        mocker.patch.object(tool, '_make_notion_request', return_value=mock_response)

        call = ToolCall(
            id="call_struct",
            name="notion_page",
            arguments={"title": "Struct Test", "content": "Content"}
        )

        result = await tool.execute(call)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "notion_page"
        assert result.tool_call_id == "call_struct"
        assert result.status == ToolStatus.SUCCESS
        assert result.error is None
        assert result.execution_time is not None
        assert result.result is not None
        assert "success" in result.result


class TestNotionToolBuildParentObject:
    """Test parent object construction."""

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_build_parent_object_page_id(self):
        """Test building parent object for page_id type."""
        tool = NotionPageTool()

        result = tool._build_parent_object(NotionParentType.PAGE_ID, "page-123")

        assert result["success"] is True
        assert result["parent"]["type"] == "page_id"
        assert result["parent"]["page_id"] == "page-123"

    @patch.dict(os.environ, {"NOTION_TOKEN": "token", "NOTION_PARENT_ID": "parent-123"})
    def test_build_parent_object_data_source_id(self):
        """Test building parent object for data_source_id type."""
        tool = NotionPageTool()

        result = tool._build_parent_object(NotionParentType.DATA_SOURCE_ID, "collection-456")

        assert result["success"] is True
        assert result["parent"]["type"] == "data_source_id"
        assert result["parent"]["data_source_id"] == "collection-456"