1# Testing Roadmap - Obelix Project

## 📊 Status Overview

### ✅ Completed

#### Phase 1: Messages Module (2025-01-06)
- [x] Test infrastructure setup (pytest, pytest-asyncio, pytest-cov, pytest-mock)
- [x] Test directory structure following PPT best practices
- [x] Message system unit tests (43 tests, all passing)
  - test_human_message.py (5 tests)
  - test_system_message.py (4 tests)
  - test_assistant_message.py (10 tests)
  - test_tool_message.py (16 tests)
  - test_standard_message.py (8 tests)

#### Phase 2: Tools Module (2025-01-06)
- [x] Tools system unit tests (108 tests, all passing)
  - test_tool_schema.py (20 tests)
  - test_tool_base.py (18 tests)
  - test_calculator_tool.py (37 tests)
  - test_notion_tool.py (33 tests)

#### Phase 3: Providers Module (2025-01-06)
- [x] Providers system unit tests (45 tests, all passing, 97% coverage)
  - test_config.py (11 tests) - GlobalConfig singleton
  - test_providers_enum.py (16 tests) - Providers enum and factory pattern
  - test_provider_registry.py (18 tests) - ProviderRegistry registration/retrieval

#### Infrastructure (2025-01-08) 🔧 **UPDATED**
- [x] Shared fixtures in conftest.py
- [x] Test suites (suite_messages.py, suite_unit.py, suite_integration.py, suite_all.py)
- [x] Documentation (README.md with detailed explanations)
- [x] Cleanup of old test files
- [x] **Fixed missing dependencies**: pytest-mock, pytest-asyncio installed
- [x] **All 196 existing tests passing** (Messages + Tools + Providers)

### 🐛 Bugs Found and Fixed
- **NotionTool Markdown Parser Bug** (2025-01-06)
  - Issue: Parser didn't handle indented markdown correctly (headers, lists)
  - Found by: `test_convert_mixed_content` in test_notion_tool.py
  - Fix: Added `.strip()` to pattern matching in `_convert_markdown_to_blocks()`
  - File: `src/tools/tool/notion_tool.py` (lines 266-285)

---

## 🎯 Next Steps - Logical Order

### 🚧 **PHASE 4: Unit Tests - Mapping Module** (IN PROGRESS - 2025-01-08)

**Status**: 🔄 **STARTED** - File created, need to fix mapping registration issues

**Progress:**
- [x] Created test file structure: `tests/unit/mapping/test_provider_mapping.py`
- [x] Created comprehensive test cases (23 tests covering all mapping functions)
- [ ] **BLOCKED**: Mapping registration not working - mappings not loaded in ProviderRegistry
- [ ] Fix mapping import/registration issue
- [ ] Run tests and verify all pass
- [ ] Add coverage reporting

**Current Issue (2025-01-08):**
```
KeyError: 'ibm_watson' - ProviderRegistry.get_mapping() fails
```

**Root Cause:** The mapping module `src/mapping/provider_mapping.py` registers mappings at import time, but tests don't import it properly.

**Next Steps:**
1. Fix mapping registration in test setup
2. Ensure mappings are loaded before running tests
3. Fix Pydantic validation issues in complex schema test
4. Verify all 23 mapping tests pass

**Test Structure Created:**
- TestIBMWatsonMapping (9 tests)
- TestOCIGenerativeAIMapping (8 tests)
- TestProviderRegistryIntegration (3 tests)
- TestEdgeCases (3 tests)

**Files to fix:**
```
tests/unit/mapping/
└── test_provider_mapping.py    # 23 tests created, needs mapping fixes
```

---

### **PHASE 5: Integration Tests**

**Files to create:**
```
tests/integration/
├── test_base_agent.py                  # Agent + Tools + Provider
├── test_tool_execution_flow.py         # Tool execution end-to-end
├── test_provider_message_conversion.py # Provider + Mapping + Messages
└── test_mcp_integration.py             # MCP tools (optional)
```

**Estimated time:** 2-3 hours

---

## 📋 Archive - Completed Phases

### **PHASE 1: Unit Tests - Tools Module** ✅ COMPLETED

**Why first?**
- Low dependencies (only depends on Messages, already tested)
- Clear responsibilities
- Easy to test in isolation

**Files to create:**
```
tests/unit/tools/
├── test_tool_schema.py        # PRIORITY 1 - Simplest, no dependencies
├── test_tool_base.py          # PRIORITY 2 - Abstract class behavior
├── test_calculator_tool.py    # PRIORITY 3 - Concrete tool, simple logic
└── test_notion_tool.py        # PRIORITY 4 - Requires API mocking
```

#### **1.1 test_tool_schema.py** (Estimated: 10 min)

**Purpose:** Test Pydantic schema generation for tools

**Test cases to implement:**
```python
class TestToolSchema:
    def test_tool_schema_creation()
        # Basic schema creation with tool_name and tool_description

    def test_tool_schema_with_required_fields()
        # Schema with required parameters

    def test_tool_schema_with_optional_fields()
        # Schema with optional parameters and defaults

    def test_tool_schema_validation_error()
        # Invalid data raises ValidationError

    def test_tool_schema_json_generation()
        # model_json_schema() generates correct JSON

    def test_tool_schema_get_tool_name()
        # Class method returns correct tool name

    def test_tool_schema_get_tool_description()
        # Class method returns correct description

    def test_tool_schema_with_nested_types()
        # Schema with Dict, List, nested Pydantic models
```

**Dependencies:** None (pure Pydantic)

**Source file:** `src/tools/tool_schema.py`

---

#### **1.2 test_tool_base.py** (Estimated: 15 min)

**Purpose:** Test ToolBase abstract class and schema creation

**Test cases to implement:**
```python
class TestToolBase:
    def test_tool_base_is_abstract()
        # Cannot instantiate ToolBase directly

    def test_tool_base_create_schema()
        # create_schema() generates MCPToolSchema

    def test_tool_base_schema_class_required()
        # Raises error if schema_class not defined

    def test_tool_base_execute_is_abstract()
        # execute() must be implemented by subclasses

    def test_mcp_tool_schema_generation()
        # Schema has name, description, inputSchema, outputSchema

    def test_tool_base_with_custom_schema()
        # Subclass with custom schema_class works correctly

    def test_tool_base_schema_input_validation()
        # inputSchema matches Pydantic model structure
```

**Dependencies:** ToolSchema (tested in 1.1)

**Source file:** `src/tools/tool_base.py`

**Notes:**
- Need to create concrete test implementation of ToolBase for testing
- Use `with pytest.raises(TypeError)` for abstract class tests

---

#### **1.3 test_calculator_tool.py** (Estimated: 20 min)

**Purpose:** Test CalculatorTool implementation (concrete tool)

**Test cases to implement:**
```python
class TestCalculatorToolSchema:
    def test_calculator_schema_generation()
        # Schema created correctly

    def test_calculator_schema_has_correct_name()
        # tool_name == "calculator"

    def test_calculator_schema_has_required_fields()
        # Schema requires: operation, a, b


class TestCalculatorToolExecution:
    """All execution tests must be async"""

    @pytest.mark.asyncio
    async def test_calculator_add()
        # 5 + 3 = 8

    @pytest.mark.asyncio
    async def test_calculator_subtract()
        # 10 - 4 = 6

    @pytest.mark.asyncio
    async def test_calculator_multiply()
        # 7 * 6 = 42

    @pytest.mark.asyncio
    async def test_calculator_divide()
        # 20 / 5 = 4

    @pytest.mark.asyncio
    async def test_calculator_divide_by_zero_error()
        # Division by 0 returns error status

    @pytest.mark.asyncio
    async def test_calculator_invalid_operation_error()
        # Unknown operation returns error

    @pytest.mark.asyncio
    async def test_calculator_missing_arguments_error()
        # Missing 'a' or 'b' returns error

    @pytest.mark.asyncio
    async def test_calculator_non_numeric_arguments()
        # String instead of number returns error

    @pytest.mark.asyncio
    async def test_calculator_returns_tool_result()
        # Result is ToolResult with correct fields

    @pytest.mark.asyncio
    async def test_calculator_execution_time_tracking()
        # execution_time field is populated
```

**Dependencies:** ToolBase, ToolSchema, ToolCall, ToolResult

**Source file:** `src/tools/tool/calculator_tool.py`

**Notes:**
- All execution tests must use `@pytest.mark.asyncio`
- Create ToolCall objects with proper structure
- Verify ToolResult status (SUCCESS/ERROR)
- Test both happy path and error cases

---

#### **1.4 test_notion_tool.py** (Estimated: 30 min)

**Purpose:** Test NotionPageTool with mocked Notion API

**Test cases to implement:**
```python
class TestNotionToolSchema:
    def test_notion_schema_generation()
    def test_notion_schema_has_correct_fields()


class TestNotionToolExecution:
    @pytest.mark.asyncio
    async def test_notion_create_page_success(mocker)
        # Mock successful API call
        # Verify page created with correct data

    @pytest.mark.asyncio
    async def test_notion_create_page_with_properties(mocker)
        # Test with custom properties

    @pytest.mark.asyncio
    async def test_notion_api_error(mocker)
        # Mock API returns error
        # Tool returns ERROR status

    @pytest.mark.asyncio
    async def test_notion_invalid_token_error(mocker)
        # Mock authentication error

    @pytest.mark.asyncio
    async def test_notion_network_timeout(mocker)
        # Mock timeout error

    @pytest.mark.asyncio
    async def test_notion_missing_required_fields(mocker)
        # Missing database_id or title
```

**Dependencies:** ToolBase, ToolSchema, pytest-mock

**Source file:** `src/tools/tool/notion_tool.py`

**Mocking strategy:**
```python
# Example mock structure
def test_notion_create_page_success(mocker):
    # Mock the Notion API client
    mock_notion = mocker.patch('src.tools.tool.notion_tool.Client')
    mock_notion.return_value.pages.create.return_value = {
        "id": "page-123",
        "url": "https://notion.so/page-123"
    }

    # Test the tool
    tool = NotionPageTool()
    result = await tool.execute(tool_call)

    # Verify
    assert result.status == ToolStatus.SUCCESS
    mock_notion.return_value.pages.create.assert_called_once()
```

**Notes:**
- Use `pytest-mock` (mocker fixture) to mock Notion API
- Don't make real API calls in unit tests
- Mock environment variables (NOTION_TOKEN, NOTION_DATABASE_ID)

---

### **PHASE 2: Unit Tests - Providers Module**

**Files to create:**
```
tests/unit/providers/
├── test_config.py              # GlobalConfig singleton
├── test_provider_registry.py   # ProviderRegistry
└── test_providers_enum.py      # Providers enum + factory
```

**Estimated time:** 1 hour

**Details:** TBD (to be defined when starting this phase)

---

### **PHASE 3: Unit Tests - Mapping Module**

**Files to create:**
```
tests/unit/mapping/
└── test_provider_mapping.py    # Tool/Message mapping per provider
```

**Estimated time:** 1 hour

**Details:** TBD

---

### **PHASE 4: Integration Tests**

**Files to create:**
```
tests/integration/
├── test_base_agent.py                  # Agent + Tools + Provider
├── test_tool_execution_flow.py         # Tool execution end-to-end
├── test_provider_message_conversion.py # Provider + Mapping + Messages
└── test_mcp_integration.py             # MCP tools (optional)
```

**Estimated time:** 2-3 hours

**Details:** TBD

---

## 🎯 Fixtures to Add to conftest.py

### For Tools Testing

```python
# Add to tests/conftest.py

# ============================================================================
# Tool Fixtures (to be added when implementing tool tests)
# ============================================================================

@pytest.fixture
def sample_calculator_tool():
    """Provides a CalculatorTool instance for testing."""
    from src.tools.tool.calculator_tool import CalculatorTool
    return CalculatorTool()


@pytest.fixture
def sample_calculator_tool_call():
    """Provides a sample ToolCall for calculator."""
    return ToolCall(
        id="call_calc_test",
        name="calculator",
        arguments={"operation": "add", "a": 10, "b": 5}
    )


@pytest.fixture
def sample_notion_tool():
    """Provides a NotionPageTool instance (requires env vars)."""
    from src.tools.tool.notion_tool import NotionPageTool
    return NotionPageTool()


@pytest.fixture
def mock_notion_api(mocker):
    """Mocks Notion API client."""
    mock_client = mocker.patch('src.tools.tool.notion_tool.Client')
    mock_client.return_value.pages.create.return_value = {
        "id": "test-page-123",
        "url": "https://notion.so/test-page-123",
        "created_time": "2024-01-01T00:00:00.000Z"
    }
    return mock_client
```

---

## 📚 Key Testing Principles (from PPT)

### Unit Tests Characteristics
- **Isolation**: Test ONE component at a time
- **No external dependencies**: Mock APIs, databases, file system
- **Fast**: Execute in milliseconds
- **Deterministic**: Same input = same output every time

### Test Organization
- **One file per class**: `test_calculator_tool.py` → only `CalculatorTool`
- **Descriptive names**: `test_calculator_divide_by_zero_error()`
- **AAA Pattern**: Arrange → Act → Assert
- **Docstrings**: Explain what each test verifies

### Async Testing
- Use `@pytest.mark.asyncio` for async functions
- Use `await` when calling async methods
- Fixtures can be async too: `@pytest.fixture(scope="function") async def ...`

---

## 🚀 Quick Start Commands

### Run specific test file
```bash
pytest tests/unit/tools/test_tool_schema.py -v
```

### Run all tools tests
```bash
pytest tests/unit/tools/ -v
```

### Run with coverage
```bash
pytest tests/unit/tools/ --cov=src/tools --cov-report=html
```

### Run single test
```bash
pytest tests/unit/tools/test_calculator_tool.py::TestCalculatorToolExecution::test_calculator_add -v
```

---

## 📝 Notes for Resuming

**Current Status (2025-01-08):**
- Messages module: ✅ Complete (43 tests, 100% coverage)
- Tools module: ✅ Complete (108 tests, high coverage)
- Providers module: ✅ Complete (45 tests, 97% coverage)
- Mapping module: 🚧 **IN PROGRESS** - 23 tests created, mapping registration issues to fix
- **Total working tests:** 196/196 ✅

**What to tell Claude when resuming:**
> "Continue testing according to tests/TESTING_ROADMAP.md. Fix the mapping registration issue in PHASE 4: test_provider_mapping.py. The mappings aren't being loaded by ProviderRegistry during tests."

**Key issue to solve:**
```python
# Need to ensure mapping module is imported before tests run
from src.mapping import provider_mapping  # This registers the mappings
```

**Important files to reference:**
- `tests/README.md` - Testing philosophy and best practices
- `tests/TESTING_ROADMAP.md` - This file
- `src/mapping/provider_mapping.py` - Code to test
- `tests/conftest.py` - Shared fixtures

---

## 📊 Test Statistics Summary

| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| Messages | 43 | ~100% | ✅ |
| Tools | 108 | ~95% | ✅ |
| Providers | 45 | 97% | ✅ |
| **Total** | **196** | **~97%** | **✅** |

---

## ✅ Checklist for Each Test File

Before considering a test file complete:

- [x] All test cases from plan implemented
- [x] Tests follow AAA pattern (Arrange, Act, Assert)
- [x] Descriptive docstrings for each test
- [x] Tests pass: `pytest <file> -v`
- [x] Code coverage checked: `pytest <file> --cov`
- [x] Edge cases covered (errors, empty values, invalid inputs)
- [x] Async tests use `@pytest.mark.asyncio`
- [x] Fixtures used where appropriate
- [x] File follows naming convention: `test_*.py`
- [x] Classes follow naming: `Test*`
- [x] Methods follow naming: `test_*`

---

**Last updated:** 2025-01-06 (17:30)
**Next action:** Create `tests/unit/mapping/test_provider_mapping.py`