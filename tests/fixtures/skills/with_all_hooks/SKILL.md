---
description: Registers all supported hooks
hooks:
  before_llm_call: "pre-llm"
  after_llm_call: "post-llm"
  before_tool_execution: "pre-tool"
  after_tool_execution: "post-tool"
  on_tool_error: "tool-error"
  before_final_response: "pre-final"
  query_end: "end"
---
Body with all hooks declared.
