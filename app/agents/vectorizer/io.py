from pydantic import BaseModel


class VectorizerOutput(BaseModel):
    embedding_id: str
    fingerprint: str

