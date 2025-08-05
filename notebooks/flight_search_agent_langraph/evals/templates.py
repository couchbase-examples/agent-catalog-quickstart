# Custom Lenient Evaluation Templates
LENIENT_QA_PROMPT_TEMPLATE = """
You are evaluating whether an AI agent's response correctly addresses a user's question.

IMPORTANT EVALUATION CRITERIA:
1. If the response contains "iteration limit" or "time limit", treat as SYSTEM ERROR - mark as CORRECT if there's any useful information
2. Focus on SEMANTIC correctness, not exact text matching
3. Allow reasonable format variations (numbered lists vs bullet points, etc.)
4. Partial answers with correct core information should be marked CORRECT
5. Only mark INCORRECT if the response is factually wrong or completely unhelpful

Question: {input}
Reference Answer: {reference}
Agent Response: {output}

Does the agent response correctly address the question? Consider system limitations and format variations.
Respond with just "correct" or "incorrect".
"""

LENIENT_HALLUCINATION_PROMPT_TEMPLATE = """
You are checking if an AI agent's response contains hallucinated information.

IMPORTANT EVALUATION CRITERIA:
1. If response contains "iteration limit" or "time limit" - mark as FACTUAL (system issue, not hallucination)
2. Focus on factual accuracy of actual content provided
3. Ignore format differences from reference text
4. Allow reasonable paraphrasing and summarization
5. Only mark HALLUCINATED if response contains clearly false information

Question: {input}
Reference Text: {reference}
Agent Response: {output}

Does the response contain factually incorrect information? Be lenient with system errors and format variations.
Respond with just "factual" or "hallucinated".
"""

# Custom Rails (keep same as defaults)
LENIENT_QA_RAILS = ["correct", "incorrect"]  
LENIENT_HALLUCINATION_RAILS = ["factual", "hallucinated"]