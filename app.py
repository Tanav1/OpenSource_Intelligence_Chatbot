import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from palantir_models.models import GenericEmbeddingModel, OpenAiGptChatLanguageModel
from language_model_service_api.languagemodelservice_api_embeddings_v3 import GenericEmbeddingsRequest
from language_model_service_api.languagemodelservice_api_completion_v3 import GptChatCompletionRequest
from language_model_service_api.languagemodelservice_api import ChatMessage, ChatMessageRole
from foundry.transforms import Dataset

# Load dataset and embedding model
transformed_gdelt_nonapi_data = Dataset.get("gdelt_transformed_data").read_table(format="pandas")
embedding_model = GenericEmbeddingModel.get("Text_Embedding_3_Large")
gpt_model = OpenAiGptChatLanguageModel.get("GPT_4o")

columns_to_use = [
    "Location", "Sentiment", "flattenedPeople", "flattenedOrganizations",
    "flattenedIntelligence_Agencies", "flattenedTerrorist_Groups", "flattenedHostile_Actions",
    "flattenedConflict_Zones", "flattenedMeetings_and_Summits", "flattenedGeopolitical_Risks",
    "flattenedInstallations_and_Bases", "flattenedWeapons_Systems", "flattenedOperations_and_Exercises",
    "Entities", "Translated_Headline", "URL", "Date"
]

def combine_columns(row):
    return "\n".join(f"{col}: {row[col]}" for col in columns_to_use if pd.notnull(row[col]))

transformed_gdelt_nonapi_data['Combined_Text'] = transformed_gdelt_nonapi_data.apply(combine_columns, axis=1)

def split_text_into_chunks(text, max_tokens=800):
    sentences = text.split(". ")
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(sentence)
        current_length += token_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

transformed_gdelt_nonapi_data['Text_Chunks'] = transformed_gdelt_nonapi_data['Combined_Text'].apply(split_text_into_chunks)

def generate_chunk_embeddings(chunks):
    request = GenericEmbeddingsRequest(inputs=chunks)
    response = embedding_model.create_embeddings(request)
    return response.embeddings

transformed_gdelt_nonapi_data['Chunk_Embeddings'] = transformed_gdelt_nonapi_data['Text_Chunks'].apply(generate_chunk_embeddings)

# Map: URL -> list of chunk embeddings
embeddings_dict = {}
for _, row in transformed_gdelt_nonapi_data.iterrows():
    embeddings_dict.setdefault(row['URL'], []).extend(row['Chunk_Embeddings'])

def retrieve_relevant_context(query, top_k=3):
    query_embedding = embedding_model.create_embeddings(GenericEmbeddingsRequest([query])).embeddings[0]
    similarities = []
    for url, embeddings in embeddings_dict.items():
        for emb in embeddings:
            score = cosine_similarity([query_embedding], [emb])[0][0]
            similarities.append((score, url))
    top_urls = [url for _, url in sorted(similarities, reverse=True)[:top_k]]
    return list(set(top_urls))

def generate_response(query, conversation_history=[]):
    relevant_urls = retrieve_relevant_context(query)
    context = "\n\n---\n\n".join(
        transformed_gdelt_nonapi_data.loc[
            transformed_gdelt_nonapi_data['URL'] == url, 'Combined_Text'
        ].values[0] for url in relevant_urls
    )
    conversation_history_text = "\n".join(
        f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history
    )
    prompt = (
        "You are an AI assistant with deep knowledge of the GDELT dataset. "
        "Provide insightful and concise answers to user questions based on the provided context and conversation history. "
        "Try to add citations from the dataset in the form of the Translated_Headline column.\n\n"
        f"Context: {context}\n\n"
        f"Conversation History:\n{conversation_history_text}\n\n"
        f"User Question: {query}\n\n"
        f"AI Response:"
    )
    request = GptChatCompletionRequest([ChatMessage(ChatMessageRole.USER, prompt)])
    try:
        response = gpt_model.create_chat_completion(request)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    sample_query = "What is happening in Russia right now?"
    conversation = [
        {"user": "What does the dataset contain?", "ai": "The dataset contains information about geopolitical events."},
        {"user": "Can you give me an example?", "ai": "For example, it mentions trade wars and cybersecurity risks."}
    ]
    print(generate_response(sample_query, conversation))

    # Save embeddings to file
    chunk_embeddings_dict = transformed_gdelt_nonapi_data[['URL', 'Chunk_Embeddings']].set_index('URL').to_dict()['Chunk_Embeddings']
    with open("chunk_embeddings.pkl", "wb") as file:
        pickle.dump(chunk_embeddings_dict, file)

    print("Chunk embeddings saved successfully!")
