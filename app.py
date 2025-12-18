from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommend import recommend
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def get_recommend(req: QueryRequest):
    recs = recommend(req.query, top_k=5)  # 5-10 as per spec
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return {"recommended_assessments": recs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)