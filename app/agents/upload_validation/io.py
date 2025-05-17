from pydantic import BaseModel
from typing import List, Any


class CleanRow(BaseModel):
    data: dict


class UploadValidationOutput(BaseModel):
    clean_rows: List[CleanRow]
    bad_rows: List[dict]

