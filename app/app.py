from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.routers import optimizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(optimizer.router)

@app.get("/")
def root():
    # redirect to /docs
    return RedirectResponse(url="/docs")
