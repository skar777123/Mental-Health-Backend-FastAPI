import cohere
import numpy as np
import warnings
from annoy import AnnoyIndex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
warnings.filterwarnings('ignore')

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# COHERE_API_KEY = "etftsr4I4QtfbZzTPMuE4AhLMDgiI0wLRdmjdWVu"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

text = """
Hello, How can I help You

My name is YourFrnd

Depression is very common thing in Human, just don't get bothered by it and socialize

Suicide is a tense mental problem

overcome on suicide is easy
"""

texts = [t.strip() for t in text.split("\n\n") if t.strip()]
texts_array = np.array(texts)

embeddings = np.array(co.embed(texts=texts).embeddings)
search_index = AnnoyIndex(embeddings.shape[1], 'angular')
for i, embed in enumerate(embeddings):
    search_index.add_item(i, embed)
search_index.build(10)

search_index.save('test.ann')

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

def search_text(query):
    query_embed = co.embed(texts=[query]).embeddings
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=False)
    return texts_array[similar_item_ids]

def ask_llm(question, num_generations=1):
    results = search_text(question)
    context = results[0] if len(results) > 0 else ""

    prompt = f"""
    More information about Mental Health:

    {context}

    Question: {question}

    Extract the answer of the question from the text provided.

    Make sure you only give required answer only
    If the text doesn't contain the answer, don't reply that the answer is not available.
    If the answer is available for the text, modify it and make it humanized and creative.
    And if the text user provided does not contain a direct answer to the question don't mention it and give an appropriate answer.
    """

    prediction = co.generate(
        prompt=prompt,
        max_tokens=70,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )

    return prediction.generations[0].text.strip()

# Define FastAPI endpoints
@app.post("/ask", response_model=QueryResponse)
def ask_question(query: QueryRequest):
    try:
        answer = ask_llm(query.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application (if needed for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
