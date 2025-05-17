from pydantic import BaseModel


class BuyerROIOutput(BaseModel):
    roi_table: dict
    narrative: str

