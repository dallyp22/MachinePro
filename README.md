# AgIQ v2: Farm Equipment Valuation System

An advanced AI-powered farm equipment valuation platform that uses cutting-edge retrieval-augmented generation (RAG) for precise agricultural machinery market value calculations.

## Features

- **Accurate Valuations**: Get fair market value estimates for farm equipment using real auction data
- **Comparable Sales Analysis**: Find and analyze similar equipment sales from auction data
- **Advanced RAG-based Retrieval**: Uses OpenAI's vector database to find the most relevant comparable sales
- **Smart Price Adjustments**: Automatically adjusts for equipment age, usage hours, and condition
- **Improved Search Coverage**: Extends search to last 180 days when fewer than 3 comparable sales are found in 90 days
- **Outlier Removal**: Uses IQR-based statistical methods to remove price outliers for more consistent valuations

## Technical Stack

- **Backend**: Python with Flask for the web server
- **AI Integration**: OpenAI's Responses API for intelligent vector store analysis
- **Data Processing**: Retrieval-augmented generation (RAG) for enhanced data extraction and analysis
- **User Interface**: Clean, responsive HTML/CSS/JS interface

## System Architecture

The system uses a multi-agent approach with specialized components:

1. **RAG Retriever**: Searches OpenAI's vector database for comparable equipment sales
2. **Valuator**: Analyzes comparable sales data and calculates fair market value with appropriate adjustments
3. **Formatter**: Structures the valuation results according to a consistent schema

## Getting Started

1. Make sure the OpenAI API key is set in your environment variables
2. Run the application using:
   ```
   gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
   ```
3. Access the web interface at http://localhost:5000

## Running Tests

Use the integrated test suite to verify all system components:
```
python test_all.py
```

## API Usage

The system provides a RESTful API endpoint at `/v2/value` that accepts POST requests with the following JSON structure:

```json
{
  "make": "John Deere",
  "model": "8370R",
  "year": 2019,
  "condition": "excellent",
  "description": "John Deere 8370R tractor with 2000 hours, excellent condition, well maintained"
}
```

The API returns a detailed valuation response including fair market value, confidence level, comparable sales, adjustments, and explanation.