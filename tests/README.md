# Test Suite Documentation

## 📁 Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest config
├── README.md                      # This file
│
├── unit/                          # Unit tests (isolated components)
│   ├── messages/
│   │   ├── test_human_message.py
│   │   ├── test_system_message.py
│   │   ├── test_assistant_message.py
│   │   ├── test_tool_message.py
│   │   └── test_standard_message.py
│   │
│   ├── tools/                     # (To be implemented)
│   ├── providers/                 # (To be implemented)
│   └── mapping/                   # (To be implemented)
│
├── integration/                   # Integration tests (To be implemented)
│
└── suites/                        # Test suites for grouped execution
    ├── suite_messages.py          # All message tests
    ├── suite_unit.py              # All unit tests
    ├── suite_integration.py       # All integration tests
    └── suite_all.py               # Complete test suite
```

---

## 🎯 Understanding Test Types

### 1️⃣ Unit Tests (`tests/unit/`)

**Definition** (PPT Slide 2-3):
> "Unit Test: verificano il corretto funzionamento di **singole funzioni o metodi**"

**Key Characteristics** (PPT Slide 3-4):
- ✅ **Complete Isolation**: Test ONE class/component at a time
- ✅ **No External Dependencies**: No database, API calls, or file system
- ✅ **Lightning Fast**: Execute in milliseconds
- ✅ **Use Mocks/Fixtures**: Fake data for any dependencies

**Example:**
```python
# tests/unit/messages/test_human_message.py
def test_human_message_creation():
    """Tests ONLY HumanMessage, isolated from everything else"""
    message = HumanMessage(content="Hello")
    assert message.content == "Hello"
    assert message.role == MessageRole.HUMAN
```

**Why separate files?**
- Each file tests **ONE CLASS ONLY**
- `test_human_message.py` → only `HumanMessage`
- `test_tool_message.py` → only `ToolCall`, `ToolResult`, `ToolMessage`
- Bug found? You know **exactly** where to look

---

### 2️⃣ Integration Tests (`tests/integration/`)

**Definition** (PPT Slide 2):
> "Integration Test: controllano che **più componenti funzionino correttamente insieme**"

**Key Characteristics:**
- ✅ Tests **interaction between 2+ components**
- ✅ May use real dependencies (or complex mocks)
- ⚠️ Slower than unit tests
- ✅ Verifies **complete workflows**

**Examples for Obelix:**

```python
# tests/integration/test_base_agent_with_tools.py
def test_base_agent_executes_calculator_tool():
    """
    Integration: BaseAgent + CalculatorTool + Provider
    Verifies the agent can:
    1. Receive a query
    2. Decide to use a tool
    3. Execute the tool
    4. Process the result
    """
    agent = BaseAgent(system_message="You are a calculator")
    agent.register_tool(CalculatorTool())

    response = agent.execute_query("Calculate 15 * 23")

    assert "345" in response.content
    assert response.tool_results is not None
```

```python
# tests/integration/test_provider_message_flow.py
def test_provider_converts_messages_correctly():
    """
    Integration: Provider + Messages + Mapping
    Verifies provider correctly converts:
    StandardMessage → Provider Format → Response
    """
    provider = OCILLm()
    messages = [
        SystemMessage(content="You are helpful"),
        HumanMessage(content="Hello")
    ]

    response = provider.invoke(messages, tools=[])

    assert isinstance(response, AssistantMessage)
    assert response.content != ""
```

---

### 3️⃣ Test Suites (`tests/suites/`)

**Definition** (PPT Slide 12-13):
> "A Test Suite allows you to **group and execute multiple test cases together**"

**Important:** These are **NOT tests** - they are **execution scripts**

**Purpose:**
- ✅ Execute organized groups of tests
- ✅ Allow selective test runs
- ✅ Useful for CI/CD pipelines

**The 4 Levels:**

```python
# suite_messages.py → Only message tests
pytest tests/unit/messages/ -v

# suite_unit.py → All unit tests
pytest tests/unit/ -v

# suite_integration.py → All integration tests
pytest tests/integration/ -v

# suite_all.py → EVERYTHING (unit + integration)
pytest tests/ -v
```

**When to use which?**

| Scenario | Suite to Use |
|----------|--------------|
| Modified only `HumanMessage` | `pytest tests/unit/messages/test_human_message.py` |
| Modified message system | `python tests/suites/suite_messages.py` |
| Modified tools | `pytest tests/unit/tools/ -v` |
| Modified BaseAgent | `pytest tests/integration/ -v` |
| Before commit | `python tests/suites/suite_all.py` |
| CI/CD pipeline | `pytest tests/ --cov=src` |

---

## 🎯 Decision Matrix: Where to Put Your Test?

### ✅ Goes in `unit/`

| Test | Why? |
|------|------|
| `HumanMessage` creation | Tests only 1 class, no dependencies |
| `ToolCall` validation | Only Pydantic validation |
| `ProviderRegistry.register()` | Only registry logic |
| `GlobalConfig.set_provider()` | Only state management |
| `ToolSchema.create()` | Only schema generation |

### ✅ Goes in `integration/`

| Test | Why? |
|------|------|
| BaseAgent executes tool | Agent + Tool + Provider together |
| Provider converts messages | Provider + Mapping + Messages |
| MasterAgent coordinates agents | Master + BaseAgent + Tools |
| MCP tool execution | MCPManager + MCPTool + Network |
| BaseAgent multi-turn conversation | Agent + Provider + Tools + Messages |

---

## 📊 Concrete Example: Testing Calculator Tool

### Unit Test (Isolated)
```python
# tests/unit/tools/test_calculator_tool.py
@pytest.mark.asyncio
async def test_calculator_tool_add():
    """Unit: tests ONLY calculation logic"""
    tool = CalculatorTool()
    tool_call = ToolCall(
        id="call_1",
        name="calculator",
        arguments={"operation": "add", "a": 5, "b": 3}
    )

    result = await tool.execute(tool_call)

    assert result.result == 8
    assert result.status == ToolStatus.SUCCESS
```

### Integration Test (With Agent)
```python
# tests/integration/test_agent_calculator.py
def test_agent_uses_calculator():
    """Integration: agent + tool + provider"""
    agent = BaseAgent(system_message="Calculate math")
    agent.register_tool(CalculatorTool())

    # Testing complete WORKFLOW
    response = agent.execute_query("What is 15 times 7?")

    assert "105" in response.content
    assert len(response.tool_results) > 0
```

---

## 🔍 Best Practices from PPT Applied

### PPT Slide 4 - Bug Tracking
> "Testare funzionalità isolate facilità il tracciamento"

✅ If `test_human_message.py` fails → bug in `HumanMessage`
✅ If `test_agent_calculator.py` fails → bug in integration

### PPT Slide 4 - Readability
> "Testare funzionalità isolate aumenta in numero i test ma ne diminuisce la complessità"

✅ 5 files with 10 tests each > 1 file with 50 tests

### PPT Slide 11 - Organization
> "Separare i test dal codice / Usare directory tests/"

✅ `src/` → production code
✅ `tests/unit/` → unit tests
✅ `tests/integration/` → integration tests

---

## ❓ Frequently Asked Questions

**Q: A test that uses `ToolBase` but mocks the tool - unit or integration?**
A: **Unit!** If you mock dependencies, it's a unit test.

**Q: Test that uses only `CalculatorTool` but calls external API?**
A: **Integration!** Even if it's one component, it uses external dependencies.

**Q: BaseAgent test with mocked provider?**
A: **Unit!** Mocked provider = no real dependencies.

**Q: BaseAgent test with real provider?**
A: **Integration!** Tests real interaction.

## 🚀 Running Tests

### Run Specific Test Files

```bash
# Test single message type
pytest tests/unit/messages/test_human_message.py -v

# Test all messages
pytest tests/unit/messages/ -v

# Test all unit tests
pytest tests/unit/ -v
```

### Run Test Suites

```bash
# Message tests only
pytest tests/suites/suite_messages.py -v
# Or: python tests/suites/suite_messages.py

# All unit tests
pytest tests/suites/suite_unit.py -v

# All integration tests
pytest tests/suites/suite_integration.py -v

# Complete test suite
pytest tests/suites/suite_all.py -v
```

### Run with Coverage

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Run by Markers

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run specific test class
pytest tests/unit/messages/test_human_message.py::TestHumanMessage -v

# Run specific test
pytest tests/unit/messages/test_human_message.py::TestHumanMessage::test_human_message_creation_with_valid_content -v
```

## 🎯 Test Organization Principles

Following best practices from testing guidelines:

### 1. **Isolation** (PPT Slide 3-4)
- Each test file tests ONE class/component
- Tests don't depend on each other
- Easy bug tracking and debugging

### 2. **Naming Convention** (PPT Slide 11)
- All test files: `test_*.py`
- All test classes: `Test*`
- All test methods: `test_*`
- Descriptive names explaining what is tested

### 3. **Fixtures** (conftest.py)
- Shared test data in `conftest.py`
- Reusable across all tests
- Follows DRY principle

### 4. **Test Suites** (PPT Slide 12-13)
- Grouped execution by domain
- Run specific subsets when needed
- Full suite for CI/CD

## 📝 Writing New Tests

### Unit Test Template

```python
"""
Unit tests for [ComponentName] class.

Tests verify:
- [Feature 1]
- [Feature 2]
- [Feature 3]
"""

import pytest
from src.module import ComponentClass


class TestComponentName:
    """Test suite for ComponentName class."""

    def test_component_basic_functionality(self):
        """Test basic component creation and usage."""
        # Arrange
        component = ComponentClass(param="value")

        # Act
        result = component.method()

        # Assert
        assert result == expected_value

    def test_component_error_handling(self):
        """Test component handles errors correctly."""
        component = ComponentClass()

        with pytest.raises(ExpectedError):
            component.invalid_operation()
```

### Using Fixtures

```python
def test_with_fixture(sample_human_message):
    """Test using a shared fixture."""
    # sample_human_message is provided by conftest.py
    assert sample_human_message.content == "What is the weather today?"
```

## 🔍 Test Coverage Goals

- **Unit Tests**: 80%+ coverage
- **Integration Tests**: Critical paths covered
- **Overall**: 70%+ coverage

## 📊 Current Status

### ✅ Completed
- Message system unit tests (5 files, ~40 tests)
- Test structure and organization
- Shared fixtures
- Test suites

### 🔄 In Progress
- Tool system tests
- Provider tests
- Mapping tests

### 📅 Planned
- Integration tests for BaseAgent
- Integration tests for MasterAgent
- End-to-end workflow tests

## 🤝 Contributing

When adding new tests:

1. **Place in correct directory**
   - Unit tests → `tests/unit/[module]/`
   - Integration tests → `tests/integration/`

2. **Follow naming conventions**
   - File: `test_[component].py`
   - Class: `Test[Component]`
   - Method: `test_[feature]_[scenario]`

3. **Add docstrings**
   - Module docstring explaining what's tested
   - Test docstrings explaining specific behavior

4. **Use fixtures when possible**
   - Check `conftest.py` for existing fixtures
   - Add new fixtures if reusable

5. **Run tests locally before committing**
   ```bash
   pytest tests/ -v
   ```

## 📚 References

- PPT Slides 3-4: Unit test characteristics and isolation
- PPT Slides 11-12: Test organization best practices
- PPT Slides 12-13: Test suites
- [Pytest Documentation](https://docs.pytest.org/)