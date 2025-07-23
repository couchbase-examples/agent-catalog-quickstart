# Hotel Support Agent - Errors and Fixes

## Error 1: agentc.trace.ApplicationSpan AttributeError

**Error:**
```
AttributeError: module 'agentc' has no attribute 'trace'
```

**Context:**
The code was trying to use `agentc.trace.ApplicationSpan()` which doesn't exist in the current Agent Catalog API.

**Fix:**
Changed from:
```python
application_span = agentc.trace.ApplicationSpan(
    "hotel-support-agent",
    tags={"version": "1.0.0", "architecture": "travel-sample"},
)
```

To:
```python
application_span = catalog.Span(name="Hotel Support Agent")
```

**Root Cause:** Using outdated or incorrect Agent Catalog API. The correct pattern is to use `catalog.Span()` method.

---

## Error 2: Capella AI Embedding Batch Size Limit

**Error:**
```
openai.UnprocessableEntityError: Error code: 422 - {'error': {'message': 'Encountered validation error for embedding request', 'type': 'model_initialization_error', 'param': {'error': 'input exceeds client batch size limit of 32', 'modelName': 'Snowflake/snowflake-arctic-embed-l-v2.0'}, 'code': 'request_initialization_error'}}
```

**Context:**
When loading 917 hotels into the vector store, the code is trying to process batches larger than 32 items, but the Capella AI embedding model `Snowflake/snowflake-arctic-embed-l-v2.0` has a batch size limit of 32.

**Current Status:** ✅ **FIXED**

**Fix:**
Reduced batch size from 50 to 25 in `data/hotel_data.py`:
```python
# Process in batches to avoid memory issues and respect Capella AI batch limit
batch_size = 25  # Well below Capella AI embedding model limit of 32
```

**Location:** 
- Error occurs in `data/hotel_data.py` line 157 in `load_hotel_data_to_couchbase()`
- Called from `main.py` line 321 in `load_hotel_data()`

---

## Error 3: Environment Variable - AGENT_CATALOG_CONN_ROOT_CERTIFICATE

**Error:**
```
ERROR:__main__:Error setting up hotel support agent: module 'agentc' has no attribute 'trace'
```

**Context:**
Agent Catalog was trying to validate the `AGENT_CATALOG_CONN_ROOT_CERTIFICATE` environment variable as a file path when it was set to an empty string.

**Fix:**
Remove the `AGENT_CATALOG_CONN_ROOT_CERTIFICATE` line from `.env` file or use:
```bash
unset AGENT_CATALOG_CONN_ROOT_CERTIFICATE
```

**Root Cause:** Environment variable was set to empty string `""` which failed Agent Catalog validation.

---

## Error 4: Outdated Prompt References

**Error:**
The `hotel_search_assistant.yaml` prompt still referenced the deleted `get_hotel_details` tool.

**Context:**
After simplifying the architecture to use only `search_vector_database`, the prompt still included references to the removed tool in:
- Tools section
- Search strategy instructions
- Example interactions

**Fix:**
Updated `prompts/hotel_search_assistant.yaml` to:
- Remove `get_hotel_details` from tools list
- Update search strategy to use only `search_vector_database`
- Modify examples to show single-tool approach
- Update instructions to reflect simplified architecture

**Root Cause:** Prompt wasn't updated after tool removal during architecture simplification.

---

## Error 5: Collection Clearing and Data Duplication

**Error:**
```
WARNING:__main__:Error clearing collection data: 'NoneType' object has no attribute 'metrics'
```

**Context:**
- DELETE query result doesn't have metadata().metrics() like other queries
- Collection clearing fails, causing data duplication (917 docs → 1834 docs)
- Data loading doesn't check if data already exists before loading

**Fix:**
1. **Fixed clear_collection_data method**:
   - Removed `result.metadata().metrics()` call
   - Added explicit execution by converting result to list
   - Added 2-second wait for deletion to propagate
   - Added verification query to check remaining count
   - Added warning if clearing is incomplete

2. **Added data existence check**:
   - Added count query before loading data
   - Skip loading if data already exists
   - Prevents duplication even if clearing fails

**Root Cause:** 
- DELETE query result structure different from other query types
- No data existence check before loading

---

## Error 6: Agent ReAct Format Parsing Issue

**Error:**
```
Invalid Format: Missing 'Action:' after 'Thought:'
```

**Context:**
- Agent was not following the proper ReAct format
- Complex prompt with too many rules was confusing the LLM
- Agent was jumping directly to Final Answer without using search_vector_database
- ReAct parser was failing to parse the agent's response correctly

**Fix:**
Simplified and standardized the prompt in `hotel_search_assistant.yaml`:
- Used exact format from working flight search agent
- Added critical rules about NEVER including Action and Final Answer in same response
- Changed from `|` to `>` content format
- Explicit instructions about response format
- Clear tool input format guidance

**Root Cause:** 
Over-complex prompt with too many rules and instructions that confused the LLM's ability to follow the required ReAct format. Also incorrect content format.

---

## Error 7: ReAct Format Parsing Failure (Final Fix)

**Error:**
```
Parsing LLM output produced both a final answer and a parse-able action
```

**Context:**
- Agent was still including both "Action:" and "Final Answer:" in the same response
- Previous fix in Error 6 was incomplete
- Agent was not following the proper template structure

**Fix:**
Updated prompt template to match working flight search agent format exactly:
- Added explicit format template showing proper `Thought:/Action:/Action Input:/Observation:` sequence
- Added comprehensive error handling rules
- Added proper template structure at end of prompt with `{tools}` and `{tool_names}` placeholders
- Added rule: "Do NOT repeat the same action more than 2 times"
- Added rule: "Present real hotel results from the search, not made-up information"

**Result:** 
✅ Agent now properly follows ReAct format and successfully uses search_vector_database tool
✅ Returns real hotel data from travel-sample database
✅ No more parsing errors

---

## Current Status - ALL ISSUES RESOLVED ✅

1. ✅ **COMPLETED:** Fixed the batch size limit issue for Capella AI embeddings
2. ✅ **COMPLETED:** Updated prompt to match simplified single-tool architecture  
3. ✅ **COMPLETED:** Fixed collection clearing and data duplication issues
4. ✅ **COMPLETED:** Fixed agent ReAct format parsing issue
5. ✅ **COMPLETED:** Agent successfully using search_vector_database tool
6. ✅ **COMPLETED:** Agent returning real hotel data from travel-sample database
7. ✅ **COMPLETED:** Test queries completing successfully
