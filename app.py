import streamlit as st
from rag_chain import build_rag_chain, query_rag
from config import cfg

st.set_page_config(
    page_title="IoT Predictive Maintenance Assistant",
    page_icon="🏭",
    layout="wide"
)

st.title("IoT Predictive Maintenance Assistant")
st.caption(f"LLM: {cfg.ollama.model} | Embeddings: {cfg.ollama.embedding} | VectorDB: ChromaDB | Framework: LangChain")
st.divider()

@st.cache_resource
def load_chain():
    with st.spinner("Loading RAG pipeline..."):
        return build_rag_chain()

chain = load_chain()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Sensor Anomaly Input")

    st.markdown("Quick Demo Scenarios:")
    demo1 = st.button("High Vibration + High Temp")
    demo2 = st.button("Pressure Drop Detected")
    demo3 = st.button("Motor Overload Alert")

    default_query = ""
    if demo1:
        default_query = "Motor-7 showing vibration 8.5 mm/s and temperature 92 degree C. What are possible causes and recommended actions?"
    elif demo2:
        default_query = "Compressor-3 showing sudden pressure drop from 180 PSI to 120 PSI. Diagnose the fault."
    elif demo3:
        default_query = "Motor-12 drawing 40 percent more current than rated load for last 10 minutes. What could be the cause?"

    query = st.text_area(
        label="Describe the anomaly or ask a maintenance question:",
        value=default_query,
        height=150,
        placeholder="e.g. Motor-7 vibration is 8.5mm/s and temperature is 92 degree C..."
    )

    submitted = st.button("Diagnose", type="primary", use_container_width=True)

with col2:
    st.subheader("Diagnosis Report")

    if submitted and query.strip():
        with st.spinner("Analysing anomaly with RAG pipeline..."):
            result = query_rag(chain, query)

        # Display answer
        st.markdown("### Diagnosis")
        st.markdown(result["answer"])

        # Display sources
        if result["sources"]:
            st.divider()
            st.markdown("### Knowledge Sources Used")
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"Source {i}: {src['source']}"):
                    st.write(f"Type: {src['doc_type']}")
                    st.write(f"File: {src['source']}")
                    st.write(f"Page: {src['page']}")

    elif submitted and not query.strip():
        st.warning("Please enter an anomaly description or question.")
    else:
        st.info("Enter a sensor anomaly on the left and click Diagnose.")

with st.sidebar:
    st.header("System Info")

    st.markdown("Model Configuration")
    st.code(f"""
LLM        : {cfg.ollama.model}
Embeddings : {cfg.ollama.embedding}
Ollama URL : {cfg.ollama.base_url}
Temperature: {cfg.ollama.temperature}
    """)

    st.markdown("Vector DB Configuration")
    st.code(f"""
DB Path    : {cfg.chroma.db_path}
Collection : {cfg.chroma.collection_name}
Top-K Docs : {cfg.retriever.top_k}
    """)

    st.divider()

    st.markdown("Knowledge Base Sources")
    st.markdown("""
- Device Manuals
- Fault Code Database
- Maintenance Logs
    """)

    st.divider()

    st.markdown("Data Folders")
    st.code(f"""
Manuals     : {cfg.data.manuals_dir}
Fault Codes : {cfg.data.fault_codes_dir}
Maint. Logs : {cfg.data.maintenance_logs_dir}
    """)

    st.divider()
    st.caption("IoT RAG Project | Kumar Gaurav")