from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.routers import optimizer, sectors
from fastapi.middleware.cors import CORSMiddleware

from app.functions.data_cleaning import setup

setup()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(optimizer.router)
app.include_router(sectors.router)

@app.get("/")
def root():
    # redirect to /docs
    return RedirectResponse(url="/docs")
