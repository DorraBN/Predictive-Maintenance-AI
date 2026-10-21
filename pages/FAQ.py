import streamlit as st
from data.preprocessing import (prepare_doc,load_datafaq,prepare_data)
from data.rag import load_embedding_model, encode_texts, retrieve_answer
st.set_page_config(page_title="FAQ",layout="wide")
st.title("FAQ")
@st.cache_data
def get_datafaq():
    return load_datafaq()

data=get_datafaq()
st.write(data.head())
data_cleaned=prepare_data(data)
st.write("data after cleaning :")
st.write(data_cleaned.head())
st.write("Ask a question about a product Maintenance.")
@st.cache_resource
def get_model():
    return load_embedding_model()

@st.cache_resource

def get_knowledge_base():
    df = prepare_data(load_datafaq())
    documents = prepare_doc(df)

    tokenizer, model = get_model()
    embeddings = encode_texts(documents, tokenizer, model)

    

    return df, documents, embeddings



try:
    with st.spinner("Loading knowledge base..."):
        df, documents, embeddings = get_knowledge_base()
        tokenizer, model = get_model()

    st.success(f"Knowledge base ready: {len(documents)} documents")

    question = st.text_input(
        "Your question",
        placeholder="Example: How can I investigate a sudden change in sensor values??"
    )

    if st.button("Get answer") and question.strip():

        results = retrieve_answer(
            question,
            documents,
            embeddings,
            tokenizer,
            model,
            top_k=3
        )

        best = results[0]

        st.subheader("Retrieved answer")
        st.write(best["document"])
        st.caption(f"Similarity score: {best['score']:.3f}")

        with st.expander("See other relevant documents"):
            for result in results[1:]:
                st.write(result["document"])
                st.caption(
                    f"Similarity score: {result['score']:.3f}"
                )

except Exception as e:
    st.error(f"An error occurred: {e}")