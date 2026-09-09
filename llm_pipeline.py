import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import openai

class LLMPipeline:
    def __init__(self, db_path="data/chroma_db", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(model_name)
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="incidents_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
    def _get_embeddings(self, texts):
        return self.embedding_model.encode(texts).tolist()

    def build_vector_db(self, df):
        print(f"Indexing {len(df)} records into Vector DB...")
        
        batch_size = 200
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            ids = [str(x) for x in batch_df.index.tolist()]
            documents = batch_df["incident_text"].tolist()
            embeddings = self._get_embeddings(documents)
            
            metadatas = batch_df[["machine_id", "incident_type", "severity", "component", "symptom", "root_cause"]].to_dict(orient="records")
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        print("Vector DB indexing complete.")

    def search_similar_incidents(self, query_text, top_k=3):
        query_emb = self._get_embeddings([query_text])
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return retrieved

    def extract_and_summarize(self, query_text, similar_incidents, llm_model="gemini-2.5-flash"):
        context_str = ""
        for idx, inc in enumerate(similar_incidents):
            meta = inc['metadata']
            context_str += f"[{idx+1}] Similar Incident on {meta['machine_id']}: '{inc['document']}' -> Root Cause: {meta['root_cause']} | Severity: {meta['severity']}\n"

        prompt = f"""You are a Factory Floor Intelligence Assistant. 
Analyze the following new incident report and extract the key information.
Then, using the historical context provided, suggest a concise root cause and action plan.

New Incident Report: "{query_text}"

Historical Context (Similar Past Incidents):
{context_str}

Output your response in valid JSON format ONLY, with the following keys:
- "machine_id": (string, extract from text if present, else "Unknown")
- "component": (string, extract from text if present)
- "symptom": (string, extract from text if present)
- "suggested_root_cause": (string, based on historical context and reasoning)
- "executive_summary": (string, 2-sentence summary of the issue and recommended next steps)
"""
        try:
            if "gemini" in llm_model:
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(
                    prompt, 
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            elif "gpt" in llm_model:
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
                response = client.chat.completions.create(
                    model=llm_model,
                    response_format={ "type": "json_object" },
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.choices[0].message.content)
            else:
                return {"error": "Unsupported model", "message": "Select gemini or gpt models."}
                
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to call LLM. Please check your API keys (e.g. OPENAI_API_KEY or GEMINI_API_KEY) and model name."
            }

if __name__ == "__main__":
    pipeline = LLMPipeline()
    if os.path.exists("data/incident_reports.csv"):
        df = pd.read_csv("data/incident_reports.csv")
        if pipeline.collection.count() == 0:
            pipeline.build_vector_db(df)
            
        query = "The conveyor belt on M-101 is vibrating excessively and sounds very loud."
        similar = pipeline.search_similar_incidents(query)
        print(f"Found {len(similar)} similar incidents.")
