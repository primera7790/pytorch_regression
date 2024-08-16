from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pages.router import router as router_pages
from operations.router import router as router_operations
from parameters.router import router as router_parameters


app = FastAPI()

app.include_router(router_pages)
app.include_router(router_operations)
app.include_router(router_parameters)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin",
                   "Authorization"],
)


