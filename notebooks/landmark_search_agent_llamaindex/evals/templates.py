"""
Custom lenient evaluation templates for Phoenix evaluators - Landmark Search Agent.

These templates are designed to be more flexible about dynamic data and focus on 
functional success rather than exact string matching.
"""

# Lenient QA evaluation template
LENIENT_QA_PROMPT_TEMPLATE = """
You are an expert evaluator assessing if an AI assistant's response correctly answers the user's question about landmarks and attractions.

FOCUS ON FUNCTIONAL SUCCESS, NOT EXACT MATCHING:
1. Did the agent provide the requested landmark information?
2. Is the core information accurate and helpful to the user?
3. Would the user be satisfied with what they received?

DYNAMIC DATA IS EXPECTED AND CORRECT:
- Landmark search results vary based on current database state
- Different search queries may return different but valid landmarks
- Order of results may vary (this is normal for search results)
- Formatting differences are acceptable

IGNORE THESE DIFFERENCES:
- Format differences, duplicate searches, system messages
- Different result ordering or landmark selection
- Reference mismatches due to dynamic search results

MARK AS CORRECT IF:
- Agent successfully found landmarks matching the request
- User received useful, accurate landmark information
- Core functionality worked as expected (search worked, results filtered properly)

MARK AS INCORRECT ONLY IF:
- Agent completely failed to provide landmark information
- Response is totally irrelevant to the landmark search request
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
You are evaluating whether an AI assistant's response about landmarks contains hallucinated (fabricated) information.

DYNAMIC DATA IS EXPECTED AND FACTUAL:
- Landmark search results are pulled from a real database
- Different searches return different valid landmarks (this is correct behavior)
- Landmark details like addresses, descriptions, and activities come from actual data
- Search result variations are normal and factual

MARK AS FACTUAL IF:
- Response contains "iteration limit" or "time limit" (system issue, not hallucination)
- Agent provides plausible landmark data from search results
- Information is consistent with typical landmark search functionality
- Results differ from reference due to dynamic search (this is expected!)

ONLY MARK AS HALLUCINATED IF:
- Response contains clearly impossible landmark information
- Agent makes up fake landmark names, addresses, or details
- Response contradicts fundamental facts about landmark search
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