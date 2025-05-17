from pydantic import BaseModel


class FeedbackOutput(BaseModel):
    drift_record: dict

