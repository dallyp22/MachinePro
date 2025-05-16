from app.agents.rag_retriever import acall as retriever_acall
from app.agents.valuator import acall as valuator_acall
from app.agents.formatter import acall as formatter_acall

async def run_chain(payload: dict) -> str:
    # 1. Retriever - get comparable sales using RAG approach
    # Create a structured query with make, model and year
    query_blob = f"{payload['make']} {payload['model']} {payload['year']} {payload['description']}"
    structured_query = f"{query_blob} make: \"{payload['make']}\" model: \"{payload['model']}\" year: \"{payload['year']}\""
    comps_json = await retriever_acall(structured_query)
    
    # Debug the comps data
    print(f"Retrieved {len(comps_json)} comparable sales")
    if comps_json and len(comps_json) > 0:
        sample_comp = comps_json[0]
        print(f"Sample comp: Sale ID: {sample_comp.get('sale_id')}, Price: ${sample_comp.get('price', 0):,.2f}")
        
        # Calculate average price from comps for debugging
        prices = [comp.get('price', 0) for comp in comps_json]
        if prices:
            avg_price = sum(prices) / len(prices)
            print(f"Average price from comps: ${avg_price:,.2f}")
            print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
            
        # Extract hours from description to help with adjustments
        import re
        hours_match = re.search(r'(\d+)\s*hours', payload.get('description', '').lower())
        if hours_match:
            hours = int(hours_match.group(1))
            print(f"Extracted hours from description: {hours}")
            # Add hours to the payload
            if 'hours' not in payload:
                payload['hours'] = hours
    
    # 2. Valuator - calculate the fair market value
    input_data = {
        "item": payload,
        "comps": comps_json
    }
    print(f"Sending valuator data with {len(comps_json)} comps...")
    valuation_raw = await valuator_acall(input_data)

    # 3. Formatter -> validated JSON string
    structured_json = await formatter_acall(valuation_raw)
    # Ensure we return a string, not None
    return structured_json if structured_json is not None else "{}"
