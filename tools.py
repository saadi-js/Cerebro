from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

tavily=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def web_search(query:str)-> str:
    """
    Search Tavily for recent, reliable sources and return the top results.
    Prioritize authoritative sources relevant to the query, such as official
    websites, reputable news outlets, research papers, reviews, travel guides,
    or primary historical sources.
    """
    results = tavily.search(query=query, max_results=5)

    output=[]
    for result in results["results"]:
        output.append(f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content'][:300]}\n")
    return "\n----\n".join(output)

@tool
def web_scrape(url:str)-> str:
    """
    Scrape the content of a webpage and return the clean text from given url for deeper reading.
    """
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style","nav","footer"]):
            script.decompose()
        text = soup.get_text(separator=" ",strip=True)[:3000]
        return text
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"








