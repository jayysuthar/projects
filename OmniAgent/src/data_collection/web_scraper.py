import requests
import logging
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebScraper:
    """
    A web scraper for collecting data from university websites.
    Specifically designed for the University of Texas at Dallas website.
    """
    
    def __init__(self, base_url: str, max_pages: int = 100, 
                 respect_robots_txt: bool = True, user_agent: str = None,
                 delay_between_requests: float = 1.0):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The base URL of the website to scrape
            max_pages: Maximum number of pages to scrape
            respect_robots_txt: Whether to respect robots.txt
            user_agent: User agent string to use
            delay_between_requests: Delay between requests in seconds
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent or "OmniAgent Scraper/0.1.0"
        self.delay = delay_between_requests
        
        self.visited_urls = set()
        self.urls_to_visit = []
        
        self.logger = logging.getLogger(__name__)
        
        # Setup headers
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and within the allowed domains."""
        parsed_url = urlparse(url)
        
        # Check if the URL belongs to UTD domain
        if not parsed_url.netloc.endswith("utdallas.edu"):
            return False
        
        # Skip URLs with query parameters (often search results or dynamic pages)
        if parsed_url.query:
            return False
        
        # Skip common non-content URLs
        excluded_patterns = [
            "/calendar/", "/login", "/search", 
            ".pdf", ".jpg", ".png", ".gif", 
            "mailto:", "javascript:", "#"
        ]
        
        for pattern in excluded_patterns:
            if pattern in url:
                return False
        
        return True
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract links from a webpage."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(current_url, href)
            
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        
        return links
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from a webpage."""
        # Extract title
        title = soup.title.text.strip() if soup.title else "No Title"
        
        # Extract main content (focus on main content areas, removing navigation, etc.)
        content_areas = soup.find_all(["article", "main", "div"], class_=["content", "main-content", "page-content"])
        
        if not content_areas:
            # If no specific content areas found, use the body
            content_areas = [soup.find("body")]
        
        # Extract text from content areas
        content_text = ""
        for area in content_areas:
            if area:
                # Remove script and style elements
                for script in area(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text
                area_text = area.get_text(separator=" ", strip=True)
                content_text += area_text + " "
        
        content_text = content_text.strip()
        
        # Return structured data
        return {
            "url": url,
            "title": title,
            "content": content_text,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status_code": 200,  # This would be the actual status code in a real request
                "content_type": "text/html"
            }
        }
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page."""
        try:
            self.logger.info(f"Scraping: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract content
            page_data = self.extract_content(soup, url)
            
            # Extract links for further scraping
            links = self.extract_links(soup, url)
            self.urls_to_visit.extend(links)
            
            # Mark as visited
            self.visited_urls.add(url)
            
            # Respect robots.txt by adding delay
            time.sleep(self.delay)
            
            return page_data
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            self.visited_urls.add(url)  # Mark as visited to avoid retrying
            return None
    
    def scrape_website(self, start_urls: List[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape the website starting from the given URLs.
        
        Args:
            start_urls: List of URLs to start scraping from
        
        Returns:
            List of dictionaries containing the scraped data
        """
        if start_urls:
            self.urls_to_visit = start_urls
        else:
            self.urls_to_visit = [self.base_url]
        
        results = []
        
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            # Get the next URL to visit
            url = self.urls_to_visit.pop(0)
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            # Scrape the page
            page_data = self.scrape_page(url)
            
            if page_data:
                results.append(page_data)
            
            self.logger.info(f"Scraped {len(self.visited_urls)} pages, {len(self.urls_to_visit)} URLs in queue")
        
        self.logger.info(f"Scraping completed. Scraped {len(results)} pages")
        return results