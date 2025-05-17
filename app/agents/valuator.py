import json
import os
from math import sqrt
from statistics import mean, stdev

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai may be unavailable
    OpenAI = None

USE_DUMMY = OpenAI is None or os.environ.get("USE_DUMMY") == "1"

def get_openai_client():
    """Get the OpenAI client with the current API key from environment."""
    if USE_DUMMY:
        raise RuntimeError("OpenAI client not available in dummy mode")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are an expert agricultural equipment appraiser specializing in auction valuations. Your task is to provide precise and data-driven valuations based strictly on verified comparable auction sales. Do not include dealer pricing, retail listings, or asking prices—only use final hammer prices from auction results. Ensure your analysis accounts for equipment condition, mileage, model year, regional demand, and recent bidding trends. Adjust valuations for depreciation, seasonality, and location-based price variations. Present your valuation in a structured format, including comparable sales, price trends, and a justified final estimate.. Input JSON has:
  item   → dict with make, model, year, condition, hours (may be null)
  comps  → list of comparable sales (price, sale_date, distance_miles)

Find auction sale prices for a {year} {make} {model} in {condition} and {description} condition, including at least three comparable auction sales from the last 90 days (i.e., since {today_minus_90_days}). Search the vector store  for final hammer prices, location, mileage, and condition. Do not include dealer pricing, retail listings, or asking prices—only use actual auction sale prices. Justify the final valuation based on bidding trends, depreciation, and mileage impact.  Show Final Valuation first. Structure output in sections with proper headings. Use bullet points for lists  takes equipment specs ({make}, {model}, {year}, {condition} and a loan principal, automatically retrieves the top‑K comparable sales from an OpenAI vector store, applies a unified exponential decay weighting combining recency and usage, computes fair‑market value plus –10%/–20% VaR floors and corresponding LTV ratios in one pass, and gives advice as to why to list this on DPA Auctions within the next month. Also provide an estimate Value if being sold on a Retail Lot.

VERY IMPORTANT: You MUST use the actual prices from the comparable sales to calculate the FMV. DO NOT use made-up values or defaults. If the comps have prices like $170,000-$225,000, your FMV should be in that range.

Algorithm:
  • Use the ACTUAL prices from comps. Calculate simple average first, then apply adjustments.
  • Age discount 1.5 % per year beyond 3 yrs.
  • Usage discount 0.03 % × ln(hours) × 100.
  • Condition multipliers: excellent +12 %, good +5 %, fair −8 %, poor −20 %.
  • Recency weight e^(−Δdays/365); Geo weight 0.8 if distance>500 mi else 1.
  • Compute weighted mean FMV; round to nearest 100.
  • Confidence = high if ≥25 comps & stdev/mean<12 %, medium if ≥10 comps, else low.

Your explanation should always be detailed and include:
  1. A summary of the comparable sales used (count, price range, average price)
  2. Specific details about key comparable sales that influenced your valuation
  3. How each adjustment factor (age, usage, condition) was applied and impacted the valuation
  4. Why this valuation is reasonable given market conditions
  5. Mention that outliers were removed if appropriate
  6. Any limitations in the data that affected confidence

Pay special attention to explaining why a LOW confidence rating was assigned when applicable, 
mentioning specifically that it's due to limited comparable sales data, but that the valuation
is still based on actual market data (not estimates).

Return EXACTLY:
{
 "fmv": <number - calculated from ACTUAL comp prices>,
 "confidence": "high|medium|low",
 "adjustments": {"age":..,"usage":..,"condition":..},
 "top3": [ {sale_id,price,sale_date,distance_miles}, … ],
 "explanation": <string>
}
"""

async def acall(data):
    """
    Function that replaces the Agent implementation
    Uses OpenAI directly to perform valuation
    """
    try:
        if USE_DUMMY:
            # Simple offline valuation using average of comp prices
            comps = data.get("comps", []) if isinstance(data, dict) else []
            prices = [c.get("price", 0) for c in comps]
            fmv = round(mean(prices), 2) if prices else 0.0
            conf = "high" if len(prices) >= 5 else "medium" if len(prices) >= 3 else "low"
            return {
                "fmv": fmv,
                "confidence": conf,
                "adjustments": {"age": 0, "usage": 0, "condition": 0},
                "top3": comps[:3],
                "explanation": "Dummy valuation using sample data",
            }

        # Get the OpenAI client with current API key
        client = get_openai_client()
        
        # Convert input data to string JSON if it's not already
        if not isinstance(data, str):
            data_str = json.dumps(data)
        else:
            data_str = data
        
        print("Valuator agent: Starting to process with Responses API...")
        
        try:
            # Using the newer Responses API
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            print("Using GPT-4o for valuation with Responses API")
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT + " Return valid JSON only."},
                    {"role": "user", "content": data_str}
                ],
                temperature=0.2,  # Low temperature for more deterministic results
            )
            
            print("Valuator agent: Received response from Responses API")
            print(f"Response keys: {dir(response)}")
            
            # Check if output_text exists
            if hasattr(response, 'output_text'):
                print("Valuator agent: Output text found")
                # Clean the output text - remove markdown formatting if present
                result = response.output_text
                # Remove markdown code block indicators (```json and ```)
                result = result.replace('```json', '').replace('```', '').strip()
                print(f"Valuator agent: Cleaned output: {result[:100]}...")
            else:
                print("Valuator agent: No output_text found, attempting to extract content...")
                # Try to extract content from other attributes
                if hasattr(response, 'content'):
                    for item in response.content:
                        if hasattr(item, 'text'):
                            # Clean up the text
                            result = item.text
                            result = result.replace('```json', '').replace('```', '').strip()
                            break
                    else:
                        # If no text found in content items
                        result = json.dumps({"error": "Could not extract text from response"})
                else:
                    # If no content attribute
                    result = json.dumps({"error": "No content found in response"})
            
        except Exception as e:
            print(f"Valuator agent: Error with Responses API: {str(e)}")
            print("Valuator agent: Falling back to Chat Completions API")
            
            # Fallback to Chat Completions API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": data_str}
                ],
                temperature=0.2
            )
            
            result = response.choices[0].message.content
        
        # Try to parse as JSON if it's not already
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError as e:
                print(f"Valuator agent: JSON parsing error: {str(e)}")
                return {
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_result": result[:500]  # Include first 500 chars of result for debugging
                }
        
        return result
        
    except Exception as e:
        print(f"Valuator agent: Critical error: {str(e)}")
        # Return a response that won't break the pipeline
        return {
            "error": f"Valuator agent failed: {str(e)}",
            "fmv": 0,
            "confidence": "low",
            "adjustments": {"age": 0, "usage": 0, "condition": 0},
            "top3": [],
            "explanation": f"Error processing valuation: {str(e)}"
        }