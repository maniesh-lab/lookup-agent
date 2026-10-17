from fastapi import FastAPI

app = FastAPI(title="lookup-agent")

@app.get("/health")
def health():
    return{"status":"ok"}