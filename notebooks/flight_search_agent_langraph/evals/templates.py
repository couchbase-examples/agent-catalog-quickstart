# Custom Lenient Evaluation Templates
LENIENT_QA_PROMPT_TEMPLATE = """
You are evaluating whether an AI agent's response correctly addresses a user's question.

FOCUS ON FUNCTIONAL SUCCESS, NOT EXACT MATCHING:
1. Did the agent provide the requested information (flights, bookings, reviews)?
2. Is the core information accurate and helpful to the user?
3. Would the user be satisfied with what they received?

DYNAMIC DATA IS EXPECTED AND CORRECT:
- Booking IDs will be DIFFERENT each time (dynamically generated - this is correct!)
- Dates like "tomorrow" are calculated dynamically (may differ from reference)
- Booking lists reflect ACTUAL session bookings (may differ from reference)
- Route sequences depend on actual booking order in this session

IGNORE THESE DIFFERENCES:
- Different booking IDs, dates, or sequences (these are dynamic!)
- Format differences, duplicate calls, system messages
- Reference mismatches due to dynamic data

MARK AS CORRECT IF:
- Agent successfully completed the action (found flights, made booking, retrieved bookings, got reviews)
- User received useful, accurate information
- Core functionality worked as expected

Question: {input}
Reference Answer: {reference}  
Agent Response: {output}

Did the agent successfully provide what the user requested, regardless of exact reference matching?
Respond with just "correct" or "incorrect".
"""

LENIENT_HALLUCINATION_PROMPT_TEMPLATE = """
You are checking if an AI agent's response contains hallucinated information.

DYNAMIC DATA IS EXPECTED AND FACTUAL:
- Booking IDs are dynamically generated (will ALWAYS be different from reference - this is correct!)
- Dates are calculated dynamically ("tomorrow", "next week" based on current date)
- Booking sequences reflect actual session bookings (not static reference data)
- Tool outputs contain real system data

MARK AS FACTUAL IF:
- Response contains "iteration limit" or "time limit" (system issue, not hallucination)
- Dynamic data differs from reference (booking IDs, dates, booking sequences)
- Agent provides plausible flight data, booking confirmations, or reviews
- Information is consistent with system capabilities

ONLY MARK AS HALLUCINATED IF:
- Response contains clearly impossible information (fake airlines, impossible routes)
- Agent makes up data it cannot access
- Response contradicts fundamental system facts

REMEMBER: Different booking IDs, dates, and sequences are EXPECTED dynamic behavior!

Question: {input}
Reference Text: {reference}
Agent Response: {output}

Does the response contain clearly false information, ignoring expected dynamic data differences?
Respond with just "factual" or "hallucinated".
"""

# Custom Rails (keep same as defaults)
LENIENT_QA_RAILS = ["correct", "incorrect"]  
LENIENT_HALLUCINATION_RAILS = ["factual", "hallucinated"]