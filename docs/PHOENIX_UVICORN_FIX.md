# Phoenix/Uvicorn Compatibility Fix for Python 3.12

## Problem

When using Arize Phoenix with Python 3.12, you may encounter this error:

```
TypeError: _patch_asyncio.<locals>.run() got an unexpected keyword argument 'loop_factory'
RuntimeWarning: coroutine 'Server.serve' was never awaited
ERROR - 💥 Phoenix failed to start
```

## Root Cause

- **Uvicorn 0.30.0+** introduced a `loop_factory` parameter in its asyncio initialization
- **nest_asyncio** (used for nested event loop support) doesn't handle this new parameter
- This breaks Phoenix server startup when using Python 3.12

## Solution

Pin uvicorn to version **0.29.x** in your `pyproject.toml`:

```toml
dependencies = [
    # ... other dependencies ...

    # CLI and utilities
    "nest-asyncio>=1.6.0,<2.0.0",

    # uvicorn version constraint for Phoenix LoopSetupType compatibility
    "uvicorn>=0.29.0,<0.30.0",
]
```

## Applied To

✅ **Landmark Search Agent** - `/notebooks/landmark_search_agent_llamaindex/pyproject.toml:46`
✅ **Flight Search Agent** - `/notebooks/flight_search_agent_langraph/pyproject.toml:27`
✅ **Hotel Search Agent** - `/notebooks/hotel_search_agent_langchain/pyproject.toml:28`

## Verification

After updating dependencies, verify the correct version is installed:

```bash
poetry show uvicorn
# or
poetry run pip show uvicorn
```

Expected output:
```
Name: uvicorn
Version: 0.29.0
```

## Alternative Solutions

If you need uvicorn 0.30.0+ for other reasons:

1. **Skip Phoenix**: Set environment variable `SKIP_PHOENIX=true`
2. **Use thread mode**: Modify Phoenix startup with `px.launch_app(run_in_thread=True)`
3. **Remove nest_asyncio**: If not needed, remove the dependency (check if LlamaIndex/LangChain require it)

## References

- [nest_asyncio compatibility issue](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/770)
- [LlamaIndex Phoenix integration issue](https://github.com/run-llama/llama_index/issues/17436)
- nest_asyncio package is archived and no longer maintained

## Last Updated

2025-10-11
