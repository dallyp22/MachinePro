"""Simple wrapper around OpenAI vector store operations."""

from typing import List, Dict
from openai import OpenAI
import os


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def create(store_name: str) -> str:
    client = get_client()
    resp = client.vector_stores.create(name=store_name)
    return resp.id


def add_embeddings(store_id: str, embeddings: List[Dict]):
    client = get_client()
    client.vector_stores.embeddings.create(vector_store_id=store_id, embeddings=embeddings)


def query(store_id: str, query_text: str, k: int = 10):
    client = get_client()
    return client.vector_stores.search(vector_store_id=store_id, query=query_text, max_num_results=k).data


def delete(store_id: str):
    client = get_client()
    client.vector_stores.delete(store_id)

