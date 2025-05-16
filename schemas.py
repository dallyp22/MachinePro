from pydantic import BaseModel, Field, constr
from typing import List, Literal

class ValuationRequest(BaseModel):
    make: str
    model: str
    year: int
    condition: str
    description: str

class ComparableSale(BaseModel):
    sale_id: str  # This will now contain "Item Name - Auction Company"
    price: float
    sale_date: str  # Format: YYYY-MM-DD
    # Optional additional fields that may be used in the frontend
    item_name: str | None = None  # Equipment name if available
    auction_company: str | None = None  # Auction company if available

class Adjustments(BaseModel):
    age: float
    usage: float
    condition: float

class ValuationResponse(BaseModel):
    fair_market_value: float = Field(..., description="USD rounded to nearest 100")
    confidence: Literal["low", "medium", "high"]
    comparable_sales: List[ComparableSale]
    adjustments: Adjustments
    explanation: str
