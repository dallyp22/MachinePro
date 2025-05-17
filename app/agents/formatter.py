import json
import os
import textwrap

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai may be missing
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

_schema = textwrap.dedent("""
{
  "type":"object",
  "properties":{
    "fair_market_value":{"type":"number"},
    "confidence":{"type":"string","enum":["low","medium","high"]},
    "comparable_sales":{"type":"array","items":{"type":"object","properties":{
      "sale_id":{"type":"string"},
      "price":{"type":"number"},
      "sale_date":{"type":"string"},
      "distance_miles":{"type":["number","null"]}
    },"required":["sale_id","price","sale_date"],"additionalProperties":false}},
    "adjustments":{"type":"object","properties":{
      "age":{"type":"number"},
      "usage":{"type":"number"},
      "condition":{"type":"number"}
    },"required":["age","usage","condition"],"additionalProperties":false},
    "explanation":{"type":"string"}
  },
  "required":["fair_market_value","confidence","comparable_sales","adjustments","explanation"],
  "additionalProperties":false
}
""")

SYSTEM_PROMPT = (
    "You take raw valuation dicts and rewrite them to conform exactly to the JSON schema "
    "below—no extra keys or trailing text.\n" + _schema
)

async def acall(data):
    """
    Function that replaces the Agent implementation
    Uses OpenAI directly to format valuation results to conform to our schema
    """
    try:
        if USE_DUMMY:
            if isinstance(data, str):
                data = json.loads(data)
            output = {
                "fair_market_value": data.get("fmv", 0),
                "confidence": data.get("confidence", "low"),
                "comparable_sales": [
                    {
                        "sale_id": c.get("sale_id"),
                        "price": c.get("price"),
                        "sale_date": c.get("sale_date"),
                        "distance_miles": c.get("distance_miles"),
                    }
                    for c in data.get("top3", [])
                ],
                "adjustments": data.get("adjustments", {"age": 0, "usage": 0, "condition": 0}),
                "explanation": data.get("explanation", ""),
            }
            return json.dumps(output)

        # Get the OpenAI client with current API key
        client = get_openai_client()
        
        # Convert input data to string JSON if it's not already
        if not isinstance(data, str):
            data_str = json.dumps(data)
        else:
            data_str = data
        
        print("Formatter agent: Starting to process with Responses API...")
        
        try:
            # Using the newer Responses API
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": """
You are a formatter agent that converts valuator results to match the required schema.
The input will be a raw JSON object that contains:
- fmv: a number representing fair market value
- confidence: "high", "medium", or "low"
- adjustments: {age, usage, condition} percentage adjustments
- top3: array of top comparable sales
- explanation: text explaining the calculation

IMPORTANT: FMV must be an actual number, not a string. Use the exact value from the input.

You must transform this into a proper JSON object with this exact schema:
```
{
  "fair_market_value": <number - same as input fmv>,
  "confidence": <string - same as input confidence>,
  "comparable_sales": [
    {
      "sale_id": <string - from input top3>,
      "price": <number - from input top3>,
      "sale_date": <string - from input top3>
    },
    ...
  ],
  "adjustments": {
    "age": <number - from input adjustments>,
    "usage": <number - from input adjustments>,
    "condition": <number - from input adjustments>
  },
  "explanation": <string - from input explanation>
}
```

FORMAT THE EXPLANATION:
1. If the confidence is "low", ensure the explanation clearly states this is due to limited comparable sales data,
   but emphasizes that the valuation is still based on actual market data, not estimates.
2. Format the explanation with appropriate paragraph breaks for readability.
3. Include key data points: number of comps used, price range, and average price.
4. When relevant, explain that outliers were removed for more accurate results.
5. Present the rationale for adjustments in a clear way.

Never alter the numeric values from the input. The fair_market_value must match the input fmv exactly.
DO NOT round or modify the values beyond what is provided in the input.
Return valid JSON only.
                    """},
                    {"role": "user", "content": data_str}
                ],
                temperature=0.1  # Low temperature for consistent results
            )
            
            print("Formatter agent: Received response from Responses API")
            print(f"Response keys: {dir(response)}")
            
            # Check if output_text exists
            if hasattr(response, 'output_text'):
                print("Formatter agent: Output text found")
                # Clean up the output text - remove markdown code block formatting if present
                output = response.output_text
                # Remove markdown code block indicators (```json and ```)
                output = output.replace('```json', '').replace('```', '').strip()
                print(f"Formatter agent: Cleaned output: {output[:100]}...")
                return output
            else:
                print("Formatter agent: No output_text found, attempting to extract content...")
                # Try to extract content from other attributes
                if hasattr(response, 'content') and getattr(response, 'content', None) is not None:
                    content_items = getattr(response, 'content', [])
                    for item in content_items:
                        if hasattr(item, 'text'):
                            # Clean up markdown formatting
                            output = item.text
                            output = output.replace('```json', '').replace('```', '').strip()
                            return output
                
                # If we can't get text, return a JSON error
                return json.dumps({"error": "Could not extract text from response"})
                
        except Exception as e:
            print(f"Formatter agent: Error with Responses API: {str(e)}")
            print("Formatter agent: Falling back to Chat Completions API")
            
            # Fallback to Chat Completions API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": data_str}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"Formatter agent: Critical error: {str(e)}")
        # Return a valid JSON error response
        return json.dumps({
            "error": f"Formatter agent failed: {str(e)}",
            "fair_market_value": 0,
            "confidence": "low",
            "comparable_sales": [],
            "adjustments": {"age": 0, "usage": 0, "condition": 0},
            "explanation": f"Error processing valuation: {str(e)}"
        })
