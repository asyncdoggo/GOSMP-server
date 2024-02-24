from fastapi import APIRouter
from pydantic import BaseModel
import json

router = APIRouter()


class SectorRequestModel(BaseModel):
    index: str


@router.post("/sectors/")
def read_sectors(request: SectorRequestModel):
    index = request.index

    if index == "nifty500":
        with open("app/data/nifty500_sectors.json") as f:
            sectors = json.load(f)

        # get unique sectors
        unique_sectors = list(set(sectors.values()))

        return {"sectors": unique_sectors}
