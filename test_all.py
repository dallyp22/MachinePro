"""
AgIQ v2 Comprehensive Test Suite
Run this file to test all components of the system
"""

import asyncio
import json
import sys
from app.agents.rag_retriever import acall as rag_retriever
from app.schemas import ValuationRequest
from app.orchestrator import run_chain

async def test_rag_retriever():
    print("\n🔍 TESTING RAG-BASED RETRIEVER")
    print("=================================")
    
    # Test with a common tractor model
    query = "John Deere 8370R tractor from 2019 in excellent condition with 2000 hours"
    make = "John Deere"
    model = "8370R"
    
    # Build a structured query
    structured_query = f"{query} make: \"{make}\" model: \"{model}\""
    
    # Call the RAG retriever
    results = await rag_retriever(structured_query)
    
    # Display results summary
    if results:
        print(f"\nFound {len(results)} comparable sales:")
        
        # Print a table header
        print(f"{'#':<3} {'Item':<25} {'Price':<12} {'Date':<12} {'Auction':<25}")
        print(f"{'-'*3} {'-'*25} {'-'*12} {'-'*12} {'-'*25}")
        
        # Print each result
        for i, result in enumerate(results[:5], 1):  # Show top 5
            print(f"{i:<3} {result['item_name'][:25]:<25} ${result['price']:>10,.2f} {result['sale_date']:<12} {result['auction_company'][:25]:<25}")
        
        if len(results) > 5:
            print(f"... and {len(results)-5} more results")
        
        # Calculate price statistics
        prices = [r['price'] for r in results]
        avg_price = sum(prices) / len(prices)
        
        print(f"\nAverage price: ${avg_price:,.2f}")
        print(f"Price range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    else:
        print("❌ No results found")
    
    print("\n=================================")
    return results

async def test_full_valuation():
    print("\n✨ TESTING FULL VALUATION PIPELINE")
    print("=================================")
    
    # Create a test request
    request = {
        "make": "John Deere",
        "model": "8370R",
        "year": 2019,
        "condition": "excellent",
        "description": "John Deere 8370R tractor with 2000 hours, excellent condition, well maintained"
    }
    
    print(f"REQUEST: {json.dumps(request, indent=2)}")
    
    # Run the full valuation pipeline
    result_json = await run_chain(request)
    result = json.loads(result_json)
    
    # Display result summary
    print(f"\nRESULT: Fair Market Value = ${result['fair_market_value']:,}")
    print(f"Confidence: {result['confidence'].upper()}")
    
    # Display adjustments
    print("\nAdjustments:")
    for adj_type, value in result['adjustments'].items():
        print(f"  - {adj_type.capitalize()}: {value}%")
    
    # Show top comps
    print(f"\nTop Comparable Sales Used:")
    for i, comp in enumerate(result['comparable_sales'][:3], 1):
        print(f"  {i}. {comp['sale_id']}")
        print(f"     Price: ${comp['price']:,}")
        print(f"     Date: {comp['sale_date']}")
    
    # Show explanation excerpt
    explanation = result['explanation']
    excerpt_length = min(300, len(explanation))
    print(f"\nExplanation Excerpt:")
    print("-" * 50)
    print(f"{explanation[:excerpt_length]}...")
    print("-" * 50)
    
    print("\n=================================")
    return result

async def run_all_tests():
    print("🧪 RUNNING ALL TESTS FOR AGIQ V2")
    print("=================================")
    
    # Try to run the retriever test
    try:
        comps = await test_rag_retriever()
        retriever_success = len(comps) > 0
        print(f"✓ Retriever Test: {'PASSED' if retriever_success else 'FAILED'}")
    except Exception as e:
        print(f"✗ Retriever Test: FAILED - {str(e)}")
        retriever_success = False
    
    # Try to run the full valuation test
    try:
        valuation = await test_full_valuation()
        valuation_success = valuation and 'fair_market_value' in valuation
        print(f"✓ Valuation Test: {'PASSED' if valuation_success else 'FAILED'}")
    except Exception as e:
        print(f"✗ Valuation Test: FAILED - {str(e)}")
        valuation_success = False
    
    # Overall result
    print("\nOVERALL RESULT:")
    if retriever_success and valuation_success:
        print("✅ All tests PASSED")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)