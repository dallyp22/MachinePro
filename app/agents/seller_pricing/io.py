from pydantic import BaseModel


class SellerPricingOutput(BaseModel):
    seller_plan: dict

