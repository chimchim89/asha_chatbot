# backend/utils/scraper.py
def scrape_events(url: str, event_type: str, location: str = None) -> list:
    # Mock response for testing
    return [{"event": f"Mock {event_type} at {url}", "location": location or "Unknown"}]
