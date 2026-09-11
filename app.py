# app.py
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="FinDocRAG - NVIDIA 10-K Assistant", page_icon="📊")

st.title("📊 FinDocRAG")
st.caption("Ask questions about NVIDIA's latest 10-K filing — answers are grounded in the actual document, with citations.")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

question = st.text_input(
    "Ask a question about NVIDIA's 10-K:",
    placeholder="e.g. What are NVIDIA's main risk factors related to competition?"
)

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant sections and generating answer..."):
        result = pipeline.ask(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Show retrieved source chunks"):
        for chunk in result["retrieved_chunks"]:
            st.markdown(f"**chunk_id {chunk['chunk_id']} | section {chunk['section']} | relevance score {chunk['rerank_score']:.2f}**")
            st.text(chunk["text"][:400] + "...")
            st.divider()

st.markdown("---")
st.caption("Built with hybrid retrieval (BM25 + vector search) + cross-encoder reranking. Grounded in SEC EDGAR filing data.")