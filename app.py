import streamlit as st
import pandas as pd
import pickle
from palantir_models.models import OpenAiGptChatLanguageModel, GenericEmbeddingModel
from language_model_service_api.languagemodelservice_api_completion_v3 import GptChatCompletionRequest
from language_model_service_api.languagemodelservice_api_embeddings_v3 import GenericEmbeddingsRequest
from language_model_service_api.languagemodelservice_api import ChatMessage, ChatMessageRole

@st.cache_data
def load_data_and_embeddings():
    dataset = pd.read_parquet("transformed_gdelt_nonapi_data.parquet")
    with open("chunk_embeddings.pkl", "rb") as file:
        embeddings_dict = pickle.load(file)
    return dataset, embeddings_dict

@st.cache_resource
def load_model():
    return OpenAiGptChatLanguageModel.get("GPT_4o")

def generate_summary(model, context):
    try:
        prompt = (
            "You are an AI assistant providing insights for an OSINT analyst. "
            "Analyze the following dataset content and generate a concise, high-level summary that highlights "
            "key entities, themes, and potential relevance for open-source intelligence purposes. "
            "Also make sure to add a list of at least 30 potential topics. "
            "Make sure to read through the entire dataset:\n\n"
            f"Dataset Content: {context}"
        )
        request = GptChatCompletionRequest([ChatMessage(ChatMessageRole.USER, prompt)])
        response = model.create_chat_completion(request)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_answer(model, query, context, conversation_history):
    try:
        conversation_history_text = "\n".join(
            [f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history]
        )
        prompt = (
            "You are an AI assistant with deep knowledge of the GDELT dataset. "
            "Provide insightful and concise answers to user questions based on the provided context and conversation history. "
            "Try to add citations from the dataset in the form of the Translated_Headline column. "
            "Make sure to read through the entire dataset before giving a response.\n\n"
            f"Context: {context}\n\n"
            f"Conversation History:\n{conversation_history_text}\n\n"
            f"User Question: {query}\n\n"
            f"AI Response:"
        )
        request = GptChatCompletionRequest([ChatMessage(ChatMessageRole.USER, prompt)])
        response = model.create_chat_completion(request)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data
def prepare_context(data, max_chars=500000):
    context = data.astype(str).fillna("").apply(lambda row: " ".join(row), axis=1).str.cat(sep=" ")
    return context[:max_chars]

st.title("Open Source Intelligence Chatbot")
st.write("Interact with open-source intelligence data in natural language.")

data, embeddings_dict = load_data_and_embeddings()
model = load_model()
initial_context = prepare_context(data)

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Dataset Summary")
with st.spinner("Generating summary..."):
    summary = generate_summary(model, initial_context)

if summary.startswith("Error"):
    st.error(summary)
else:
    st.markdown('## Key ideas and themes of the dataset:')
    st.markdown(summary)

st.subheader("Chat Interface")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

for entry in st.session_state.conversation_history:
    if isinstance(entry, dict) and "role" in entry and "content" in entry:
        if entry["role"] == "user":
            st.markdown(f"**User:** {entry['content']}")
        elif entry["role"] == "ai":
            st.markdown(f"**AI:** {entry['content']}")

st.divider()
st.subheader("Ask Your Question")
question = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question!")
    else:
        response = get_answer(
            model=model,
            query=question,
            context=initial_context,
            conversation_history=st.session_state.conversation_history
        )
        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "ai", "content": response})
        st.markdown(f"**AI:** {response}")
