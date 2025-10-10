"""
Custom lenient evaluation templates for Phoenix evaluators.

These templates are designed to be more flexible about dynamic data and focus on 
functional success rather than exact string matching.
"""

# Lenient QA evaluation template - copied from proven flight agent pattern
LENIENT_QA_PROMPT_TEMPLATE = """
You are evaluating whether an AI agent's response correctly addresses a user's question.

FOCUS ON FUNCTIONAL SUCCESS, NOT EXACT MATCHING:
1. Did the agent provide the requested information (hotels, bookings, reviews)?
2. Is the core information accurate and helpful to the user?
3. Would the user be satisfied with what they received?

DYNAMIC DATA IS EXPECTED AND CORRECT:
- Hotel search results will be DIFFERENT each time (dynamically searched - this is correct!)
- Hotel listings reflect ACTUAL database content (may differ from reference)
- Search results depend on vector similarity matching
- Hotel details come from real travel-sample data

IGNORE THESE DIFFERENCES:
- Different hotel results, search order, or sequences (these are dynamic!)
- Format differences, duplicate calls, system messages
- Reference mismatches due to dynamic search data

MARK AS CORRECT IF:
- Agent successfully completed the action (found hotels, provided search results, retrieved data)
- User received useful, accurate hotel information
- Core functionality worked as expected

Question: {input}
Reference Answer: {reference}
Agent Response: {output}

Did the agent successfully provide what the user requested, regardless of exact reference matching?
Respond with just "correct" or "incorrect".
"""

# Lenient hallucination evaluation template - copied from proven flight agent pattern
LENIENT_HALLUCINATION_PROMPT_TEMPLATE = """
You are checking if an AI agent's response contains hallucinated information.

DYNAMIC DATA IS EXPECTED AND FACTUAL:
- Hotel search results are dynamically retrieved (will ALWAYS be different from reference - this is correct!)
- Hotel details come from real travel-sample database
- Search results reflect actual vector similarity matching
- Tool outputs contain real system data

MARK AS FACTUAL IF:
- Response contains "iteration limit" or "time limit" (system issue, not hallucination)
- Dynamic hotel data differs from reference (different hotels found)
- Agent provides plausible hotel information, search results, or database content
- Information is consistent with system capabilities

ONLY MARK AS HALLUCINATED IF:
- Response contains clearly impossible information (fake hotels, impossible locations)
- Agent makes up data it cannot access
- Response contradicts fundamental system facts

REMEMBER: Different hotel search results are EXPECTED dynamic behavior!

Question: {input}
Reference Text: {reference}
Agent Response: {output}

Does the response contain clearly false information, ignoring expected dynamic data differences?
Respond with just "factual" or "hallucinated".
"""

# Lenient evaluation rails (classification options)
LENIENT_QA_RAILS = ["correct", "incorrect"]
LENIENT_HALLUCINATION_RAILS = ["factual", "hallucinated"]