import re

from Agents import writer_chain, critic_chain
from tools import web_search, web_scrape


def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        )

    return str(content)


def extract_urls(search_results: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^\s)>\"]+", search_results)))

def run_research_pipeline(topic : str) -> dict:

    state={}

    # Step 1: Search for relevant sources
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    state["search_results"] = extract_text(web_search.invoke(topic))
    urls = extract_urls(state["search_results"])
    if len(urls) < 3:
        raise RuntimeError(
            f"Tavily returned only {len(urls)} usable URLs; at least 3 are required."
        )

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    scraped_sources = []
    for index, url in enumerate(urls[:3], start=1):
        scraped_content = extract_text(web_scrape.invoke({"url": url}))
        scraped_sources.append(
            f"Source {index}\nURL: {url}\nContent:\n{scraped_content}"
        )
    state["reader_result"] = "\n\n---\n\n".join(scraped_sources)

    print("\n reader result ",state['reader_result'])

    # Step 3: Writer Chain
    print("\n"+" ="*50)
    print("step 3 - Writer chain is generating a research report ...")
    print("="*50)


    research_combined= (
        f"SEARCH_RESULTS: \n{state['search_results']}\n\n"
        f"READER_RESULT: \n{state['reader_result']}\n\n"
        
        )

    

    state["research_report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })

    print("\n Research Report ",state['research_report'])


    # Step 4: Critic Chain
    print("\n"+" ="*50)
    print("step 4 - Critic chain is evaluating the research report ...")
    print("="*50)

    state["critic_feedback"] = critic_chain.invoke({
        "report": state['research_report']})

    print("\n Critic Feedback ",state['critic_feedback'])


    return state

if __name__ == "__main__":
    topic=input("Enter a research topic: ")
    run_research_pipeline(topic)
    



    