import arxiv
import datetime
import logging
import asyncio
from typing import List, Dict
from llm import ArxivQA

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

arxiv_qa = ArxivQA()

def fetch_papers() -> List[arxiv.Result]:
    """Fetch papers from arXiv for the cs.CL category published today."""
    # today = datetime.date.today()
    today = datetime.date(2024,7,8)
    query = f"cat:cs.CL"
    
    search = arxiv.Search(
        query=query,
        max_results=150,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = list(search.results())
    logging.info(f"Fetched {len(papers)} papers from arXiv")
    return papers

async def process_paper(paper: arxiv.Result) -> Dict:
    try:
        query = f"Paper: {paper.title}, Summary: {paper.summary}"
        response = await arxiv_qa.acall(query)
        
        return {
            "title": paper.title,
            "authors": ", ".join(author.name for author in paper.authors),
            "url": paper.pdf_url,
            "brief_summary": response.data.brief,
            "potential_applications": response.data.potential_applications,
        }
    except Exception as e:
        logging.error(f"Error processing paper {paper.title}: {str(e)}")
        return None

def generate_newsletter(papers: List[Dict]) -> str:
    """Generate a markdown table from processed papers."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    newsletter = f"# arXiv CS.CL Newsletter for {today}\n\n"
    
    # Table header
    newsletter += "| Title | Authors | Brief Summary | Potential Applications | Link |\n"
    newsletter += "|-------|---------|---------------|------------------------|------|\n"
    
    for paper in papers:
        if paper:  # Check if paper is not None
            title = paper['title'].replace('|', '&#124;')
            authors = paper['authors'].replace('|', '&#124;')
            brief_summary = paper['brief_summary'].replace('\n', ' ').replace('|', '&#124;')
            potential_applications = paper['potential_applications'].replace('\n', ' ').replace('|', '&#124;')
            url = paper['url']
            
            newsletter += f"| {title} | {authors} | {brief_summary} | {potential_applications} | [Link]({url}) |\n"
    
    return newsletter

async def main():
    try:
        papers = fetch_papers()
        processed_papers = await asyncio.gather(*[process_paper(paper) for paper in papers])
        processed_papers = [p for p in processed_papers if p]  # Remove None values
        
        if not processed_papers:
            logging.warning("No papers were successfully processed.")
            return

        newsletter_content = generate_newsletter(processed_papers)
        
        output_file = f"{datetime.date.today().strftime('%Y-%m-%d')}_arxiv_cs_cl_newsletter.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(newsletter_content)
        
        logging.info(f"Newsletter generated successfully: {output_file}")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())