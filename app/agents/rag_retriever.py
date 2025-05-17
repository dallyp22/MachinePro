"""
Enhanced RAG-based retriever for farm equipment valuation
Uses OpenAI's vector store with improved metadata extraction
"""

import os
import re
import json
import datetime
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Union

def get_openai_client():
    """Get the OpenAI client with the current API key from environment"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)

def get_vector_store_id():
    """Get the Vector Store ID from environment"""
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        raise ValueError("OPENAI_VECTOR_STORE_ID environment variable is not set")
    return vector_store_id

def enhance_search_query(original_query: str) -> str:
    """
    Enhance the original query to be more effective for auction sales search
    """
    # Add context to make the query more specific to farm equipment auction sales
    return f"Searching for comparable farm equipment sales: {original_query}"

def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize dollar amounts
    text = re.sub(r'\$\s*(\d+),(\d+)', r'$\1\2', text)
    return text

def extract_date(text: str) -> Optional[str]:
    """Extract date in ISO format (YYYY-MM-DD) from text"""
    # Try different date formats
    patterns = [
        # MM/DD/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # MM-DD-YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
        # YYYY/MM/DD
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
        # DD/MM/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
        # Textual format like "January 15, 2023"
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})',
         lambda m: f"{m.group(3)}-{['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'].index(m.group(1))+1:02d}-{int(m.group(2)):02d}"),
    ]
    
    for pattern, formatter in patterns:
        matches = re.search(pattern, text)
        if matches:
            try:
                return formatter(matches)
            except:
                continue
    
    # If we couldn't extract a date, return None
    return None

def extract_prices(text: str) -> List[float]:
    """Extract all valid prices from text"""
    all_prices = []
    
    # Match price patterns like $45,000 or $45000 or 45,000 USD
    price_patterns = [
        r'\$\s*([\d,]+)(?:\.\d+)?',  # $45,000 or $45000
        r'([\d,]+)(?:\.\d+)?\s*dollars',  # 45,000 dollars
        r'([\d,]+)(?:\.\d+)?\s*USD',  # 45,000 USD
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Clean the match and convert to float
                price_str = match.replace(',', '')
                price = float(price_str)
                
                # Only include reasonable equipment prices (between $1K and $1M)
                if 1000 <= price <= 1000000:
                    all_prices.append(price)
            except:
                continue
    
    return all_prices

def extract_equipment_brand_and_model(text: str, make_hint: Optional[str] = None) -> Tuple[str, str]:
    """
    Extract equipment brand and model from text.
    Use make_hint to prioritize a specific brand if provided.
    """
    # Major farm equipment brands to look for
    brands = [
        "John Deere", "JOHN DEERE",
        "Case IH", "CASE IH", "CASE", "Case",
        "New Holland", "NEW HOLLAND",
        "Kubota", "KUBOTA",
        "Massey Ferguson", "MASSEY FERGUSON",
        "AGCO", "Fendt", "FENDT",
        "Claas", "CLAAS",
        "DEUTZ-FAHR", "Deutz-Fahr",
        "Caterpillar", "CAT"
    ]
    
    # First try to match the make_hint if provided
    detected_brand = "Unknown"
    if make_hint:
        make_upper = make_hint.upper()
        for brand in brands:
            if brand.upper() in make_upper or make_upper in brand.upper():
                detected_brand = brand
                break
    
    # If we didn't match with hint, try to find in text
    if detected_brand == "Unknown":
        for brand in brands:
            if brand in text:
                detected_brand = brand
                break
    
    # Normalize brand name
    if "JOHN DEERE" in detected_brand.upper():
        detected_brand = "John Deere"
    elif "CASE" in detected_brand.upper():
        detected_brand = "Case"
    elif "NEW HOLLAND" in detected_brand.upper():
        detected_brand = "New Holland"
    elif "KUBOTA" in detected_brand.upper():
        detected_brand = "Kubota"
    elif "MASSEY" in detected_brand.upper():
        detected_brand = "Massey Ferguson"
    
    # Extract model number - different patterns for different brands
    model = "Unknown Model"
    
    # John Deere pattern: digit followed by letter(s) and optional R (e.g., 8370R, 7R)
    if "John Deere" in detected_brand:
        jd_patterns = [
            r'(\d+[A-Z]+R?)',  # 8370R, 5075E
            r'([A-Z]\d+[A-Z]?)',  # S780, X9
        ]
        for pattern in jd_patterns:
            matches = re.search(pattern, text)
            if matches:
                model = matches.group(1)
                break
    
    # Case pattern
    elif "Case" in detected_brand:
        case_patterns = [
            r'(\d+[A-Z]+)',  # CASE 4440
            r'([A-Z]+-\d+)',  # CVX-175
        ]
        for pattern in case_patterns:
            matches = re.search(pattern, text)
            if matches:
                model = matches.group(1)
                break
    
    # New Holland pattern
    elif "New Holland" in detected_brand:
        nh_patterns = [
            r'(T\d+\.\d+)',  # T6.175
            r'(T\d+)',  # T7, T8
        ]
        for pattern in nh_patterns:
            matches = re.search(pattern, text)
            if matches:
                model = matches.group(1)
                break
    
    return detected_brand, model

def extract_auction_company(text: str) -> str:
    """Extract auction company name from text"""
    # Common auction company name patterns
    patterns = [
        r'([A-Z][A-Z\s&]+AUCTION)',  # SMITH AUCTION
        r'([A-Z][A-Z\s&]+AUCTIONEERS)',  # JONES AUCTIONEERS
        r'AUCTION[S]?:\s*([A-Za-z\s&]+)',  # AUCTIONS: Smith & Co
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text)
        if matches:
            return matches.group(1).strip()
    
    # If we don't find a specific company, look for location
    location_pattern = r'(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z]{2})'  # in Chicago, IL
    location_matches = re.search(location_pattern, text)
    if location_matches:
        return f"{location_matches.group(1)}, {location_matches.group(2)}"
    
    # Default
    return "Unknown Auction"

def search_with_rag(search_query: str, make: Optional[str] = None, model: Optional[str] = None, year: Optional[int] = None, k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform RAG-based search for comparable farm equipment sales
    
    Args:
        search_query: The search query text
        make: Equipment manufacturer (e.g., "John Deere")
        model: Equipment model (e.g., "8370R")
        year: Equipment year (e.g., 2020)
        k: Maximum number of results to return
        
    Returns:
        List of comparable sales with metadata
    """
    try:
        # Get OpenAI client and vector store ID
        client = get_openai_client()
        vstore_id = get_vector_store_id()
        
        # Enhance query with farm equipment context
        enhanced_query = enhance_search_query(search_query)
        print(f"Original query: {search_query}")
        print(f"Enhanced query: {enhanced_query}")
        
        # Perform vector store search
        print(f"Attempting to search vector store with ID: {vstore_id}")
        results = client.vector_stores.search(
            vector_store_id=vstore_id,
            query=enhanced_query,
            max_num_results=k,
            rewrite_query=True,
        ).data
        print(f"Vector store search succeeded, got {len(results)} results")
        
        # Process the results
        serializable_results = []
        for i, result in enumerate(results):
            # Extract content from the result
            content = ""
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                else:
                    content = str(result.content)
            elif hasattr(result, 'text'):
                content = getattr(result, 'text', '')
            
            if not content:
                print(f"WARNING: No content found in result {i}")
                continue
            
            # Clean and normalize the content
            clean_content = clean_and_normalize_text(content)
            truncated_content = clean_content[:1000]  # Truncate for display
            
            # Extract sale information
            all_prices = extract_prices(clean_content)
            print(f"Extracted data from result {i}:")
            
            # Extract date
            extracted_date = extract_date(clean_content)
            if not extracted_date:
                # Default to a recent date if we can't extract one
                extracted_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            print(f"  - Date: {extracted_date}")
            
            # Extract brand and model, using the provided make as a hint
            brand, model_name = extract_equipment_brand_and_model(clean_content, make)
            
            # Extract auction company
            auction = extract_auction_company(clean_content)
            print(f"  - Auction: {auction}")
            
            # Get the average price from all prices found
            print(f"  - All prices found: {all_prices}")
            avg_price = sum(all_prices) / len(all_prices) if all_prices else 0.0
            
            # If we don't have a reasonable price, skip this result
            if avg_price < 1000:
                print(f"  - Skipping result {i} due to unreasonable price: ${avg_price:.2f}")
                continue
            
            # Create a sale ID that combines the item name and auction company
            item_name = f"{brand} {model_name}"
            sale_id = f"{item_name} - {auction}"
            
            # Create a result object with all the extracted information
            item = {
                "sale_id": sale_id,
                "item_name": item_name,
                "auction_company": auction,
                "price": avg_price,
                "sale_date": extracted_date,
                "text": truncated_content
            }
            serializable_results.append(item)
        
        # Filter and process results based on recency
        now = datetime.now()
        cutoff_90_days = now - timedelta(days=90)
        cutoff_180_days = now - timedelta(days=180)
        
        # First, try to get sales from last 90 days
        recent_90_day_results = []
        recent_180_day_results = []
        
        for item in serializable_results:
            try:
                # Try to parse the date string
                sale_date = datetime.strptime(item['sale_date'], "%Y-%m-%d")
                
                # Check if the sale is recent (within 90 days)
                if sale_date >= cutoff_90_days:
                    recent_90_day_results.append(item)
                # Check if within 180 days as a backup
                elif sale_date >= cutoff_180_days:
                    recent_180_day_results.append(item)
            except Exception as e:
                # If date parsing fails, include the item anyway but in the 180-day bucket
                recent_180_day_results.append(item)
        
        print(f"Filtered to {len(recent_90_day_results)} sales within the last 90 days")
        
        # If we have fewer than 3 results in the last 90 days, extend to 180 days
        if len(recent_90_day_results) < 3:
            print(f"Fewer than 3 recent sales found, extending search to 180 days...")
            print(f"Found {len(recent_180_day_results)} additional sales from 90-180 days ago")
            recent_results = recent_90_day_results + recent_180_day_results
            print(f"Using {len(recent_results)} sales from the last 180 days")
        else:
            recent_results = recent_90_day_results
        
        # If still not enough results, use all available data
        if len(recent_results) < 3:
            print("Still fewer than 3 sales, using all available sales")
            recent_results = serializable_results
        
        # Remove outliers if we have enough data points (at least 5)
        if len(recent_results) >= 5:
            print("Removing price outliers...")
            prices = [item['price'] for item in recent_results]
            q1 = np.percentile(prices, 25)
            q3 = np.percentile(prices, 75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Filter out outliers
            filtered_results = [item for item in recent_results 
                                if lower_bound <= item['price'] <= upper_bound]
            
            removed_count = len(recent_results) - len(filtered_results)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers outside the range ${lower_bound:,.2f}-${upper_bound:,.2f}")
                if len(filtered_results) >= 3:  # Only use filtered results if we still have enough
                    recent_results = filtered_results
                else:
                    print(f"Too many outliers removed, reverting to original set to maintain minimum data points")
            else:
                print("No outliers found")
        
        print(f"Final dataset has {len(recent_results)} comparable sales")
        return recent_results
    
    except Exception as e:
        print(f"Error in RAG-based search: {str(e)}")
        # Return an empty list in case of error
        return []

async def acall(query_text):
    """
    Function that replaces the Agent implementation
    Uses RAG approach to retrieve comparable sales
    """
    print(f"Retriever agent: starting search with query: {query_text[:50]}...")
    
    # Extract structured data from query text using regex
    make_pattern = r'make:\s*["\'"]?([^"\'",]+)["\'"]?'
    model_pattern = r'model:\s*["\'"]?([^"\'",]+)["\'"]?'
    year_pattern = r'year:\s*["\'"]?(\d{4})["\'"]?'
    
    make_match = re.search(make_pattern, query_text)
    model_match = re.search(model_pattern, query_text)
    year_match = re.search(year_pattern, query_text)
    
    make = make_match.group(1) if make_match else None
    model = model_match.group(1) if model_match else None
    year = int(year_match.group(1)) if year_match else None
    
    # Perform RAG-based search
    results = search_with_rag(query_text, make=make, model=model, year=year)
    print(f"Retriever agent: search completed, found {len(results)} results")
    
    if results and len(results) > 0:
        print(f"Retriever agent: first result: {json.dumps(results[0], indent=2)}")
    
    # Calculate some statistics about the results for logging
    if results:
        print(f"Retrieved {len(results)} comparable sales")
        print(f"Sample comp: Sale ID: {results[0]['sale_id']}, Price: ${results[0]['price']:,.2f}")
        
        # Calculate average and range
        prices = [r['price'] for r in results]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        
        print(f"Average price from comps: ${avg_price:,.2f}")
        print(f"Price range: ${min_price:,.2f} - ${max_price:,.2f}")
    else:
        print("No comparable sales found")
    
    return results