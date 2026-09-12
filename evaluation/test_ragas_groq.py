# evaluation/test_ragas_groq.py
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

load_dotenv()

# Point the OpenAI client at Groq's OpenAI-compatible endpoint
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

llm = llm_factory("openai/gpt-oss-20b", client=client)

scorer = Faithfulness(llm=llm)

async def main():
    result = await scorer.ascore(
        user_input="Where is NVIDIA's headquarters?",
        response="NVIDIA's headquarters is in Santa Clara, California.",
        retrieved_contexts=[
            "Item 2. Properties. Our headquarters is in Santa Clara, California. We own and lease approximately 3 million square feet of building space."
        ]
    )
    print(f"Faithfulness score: {result.value}")

if __name__ == "__main__":
    asyncio.run(main())