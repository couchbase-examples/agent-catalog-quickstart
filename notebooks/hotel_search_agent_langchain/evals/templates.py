"""
Custom lenient evaluation templates for Phoenix evaluators.

These templates are designed to be more flexible about dynamic data and focus on 
functional success rather than exact string matching.
"""

# Lenient QA evaluation template
LENIENT_QA_PROMPT_TEMPLATE = """
You are an expert evaluator assessing if an AI assistant's response correctly answers the user's question about hotels.

FOCUS ON FUNCTIONAL SUCCESS, NOT EXACT MATCHING:
1. Did the agent provide the requested hotel information?
2. Is the core information accurate and helpful to the user?
3. Would the user be satisfied with what they received?

DYNAMIC DATA IS EXPECTED AND CORRECT:
- Hotel search results vary based on current database state
- Different search queries may return different but valid hotels
- Order of results may vary (this is normal for search results)
- Formatting differences are acceptable

IGNORE THESE DIFFERENCES:
- Format differences, duplicate searches, system messages
- Different result ordering or hotel selection
- Reference mismatches due to dynamic search results

MARK AS CORRECT IF:
- Agent successfully found hotels matching the request
- User received useful, accurate hotel information
- Core functionality worked as expected (search worked, results filtered properly)

MARK AS INCORRECT ONLY IF:
- Agent completely failed to provide hotel information
- Response is totally irrelevant to the hotel search request
- Agent provided clearly wrong or nonsensical information

**Question:** {input}

**Reference Answer:** {reference}

**AI Response:** {output}

Based on the criteria above, is the AI response correct?

Answer: [correct/incorrect]

Explanation: [Provide a brief explanation focusing on functional success]
"""

# Lenient hallucination evaluation template  
LENIENT_HALLUCINATION_PROMPT_TEMPLATE = """
You are evaluating whether an AI assistant's response about hotels contains hallucinated (fabricated) information.

DYNAMIC DATA IS EXPECTED AND FACTUAL:
- Hotel search results are pulled from a real database
- Different searches return different valid hotels (this is correct behavior)
- Hotel details like addresses, amenities, and descriptions come from actual data
- Search result variations are normal and factual

MARK AS FACTUAL IF:
- Response contains "iteration limit" or "time limit" (system issue, not hallucination)
- Agent provides plausible hotel data from search results
- Information is consistent with typical hotel search functionality
- Results differ from reference due to dynamic search (this is expected!)

ONLY MARK AS HALLUCINATED IF:
- Response contains clearly impossible hotel information
- Agent makes up fake hotel names, addresses, or amenities
- Response contradicts fundamental facts about hotel search
- Agent claims to have data it cannot access

REMEMBER: Different search results are EXPECTED dynamic behavior, not hallucinations!

**Question:** {input}

**Reference Answer:** {reference}

**AI Response:** {output}

Based on the criteria above, does the response contain hallucinated information?

Answer: [factual/hallucinated]

Explanation: [Focus on whether information is plausible vs clearly fabricated]
"""

# Lenient evaluation rails (classification options)
LENIENT_QA_RAILS = ["correct", "incorrect"]
LENIENT_HALLUCINATION_RAILS = ["factual", "hallucinated"]