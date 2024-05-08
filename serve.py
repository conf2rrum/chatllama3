from fastapi import FastAPI
from langserve import add_routes
from chain import rag_chain

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces"
)

add_routes(
    app,
    rag_chain,
    path="/ask"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)