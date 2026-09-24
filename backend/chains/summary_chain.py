from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import OLLAMA_HOST, OLLAMA_MODEL

# Define the scholarship summary prompt template
SUMMARY_PROMPT_TEMPLATE = """You are an AI assistant helping college students understand scholarships.
Explain the following scholarship in exactly one clear, friendly, plain-English sentence for a student:

Name: {name}
Target Category: {category}
Minimum CGPA: {min_cgpa}
Maximum Family Income: {max_family_income}
Eligible Academic Years: {eligible_years}
Scholarship Amount: {amount}
Application Deadline: {deadline}
Required Documents: {required_docs}

Return ONLY the single sentence summary with no conversational filler, bullet points, quotes, or preamble."""

summary_prompt = PromptTemplate(
    input_variables=[
        "name",
        "category",
        "min_cgpa",
        "max_family_income",
        "eligible_years",
        "amount",
        "deadline",
        "required_docs",
    ],
    template=SUMMARY_PROMPT_TEMPLATE,
)

# Initialize ChatOllama with configured host and model
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST,
    temperature=0.3,
)

# Build the LCEL summary chain
summary_chain = summary_prompt | llm | StrOutputParser()
