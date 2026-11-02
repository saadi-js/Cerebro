from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, web_scrape
import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-flash-lite-latest",
    temperature=0,
)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[web_scrape]
    )

# Writer Chain


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research writer.
Use only evidence from the gathered research. Do not invent statistics, claims,
or sources. Cite claims with [1], [2], etc., and include a Sources section
listing the title and URL for every cited source. If evidence is insufficient,
say so explicitly."""),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (numbered titles and URLs matching the inline citations)

Be detailed, factual, professional, and distinguish evidence from interpretation."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic Chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.
Check whether major claims are supported by inline citations, whether cited
sources support the claims, whether statistics include dates and attribution,
and whether the report distinguishes evidence from interpretation.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
