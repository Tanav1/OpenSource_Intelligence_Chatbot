# Open Source Intelligence Q&A Bot (RAG)

## Overview

The Open Source Intelligence Q&A Bot is a Retrieval-Augmented Generation (RAG) system designed to assist analysts in querying and understanding open-source geopolitical data. This system allows users to ask natural language questions and receive context-aware, LLM-generated responses grounded in structured data.

This project was developed as part of the Palantir Winter Defense Tech Fellowship, where it was recognized among the top 3% of submissions and selected for presentation at Palantir's Washington, D.C. office. This project uses precomputed embeddings to identify relevant segments from the dataset and feeds them into GPT-4o to answer user queries or generate dataset-wide summaries. This ensures that the responses are not just fluent and coherent but also **factually grounded in the data**.

## Features

- Generate high-level dataset summaries highlighting key geopolitical themes and entities.
- Ask natural language questions and receive grounded, context-aware answers.
- View citation-style references derived from original headline sources.
- Access a conversational interface that preserves chat history and provides continuity.
- Leverage precomputed vector embeddings for real-time contextual retrieval.

## Technical Stack

- **LLM Provider**: GPT-4o
- **Frontend**: Streamlit
- **Embedding Model**: Palantir’s internal embedding services
- **Platform**: Palantir Foundry (required for full functionality)
- **Data**: GDELT threat-related datasets

## Data Pipeline

A data pipeline was developed within Palantir Foundry to transform and enrich the GDELT dataset. The pipeline includes the following stages:

1. Union of multiple threat-specific datasets (e.g., Russia, Iran, ISIS, Hamas).
2. Translation of non-English headlines to English.
3. Extraction of named entities from headlines.
4. Expansion of structured fields into individual columns.
5. Sentiment analysis on translated headlines.
6. Classification of headline origin.
7. Generation of document-level embeddings.

### Pipeline Diagram

![Data Pipeline Screenshot](.pipeline.png)

## Demo

To view a walkthrough and demo of the system in action, visit the following video link:  
[YouTube Demo Video]([https://studio.youtube.com/video/79PtH75LJiU/edit](https://www.youtube.com/watch?v=79PtH75LJiU))

## Important Note

This code is designed to run exclusively within the Palantir Foundry platform. Due to the use of proprietary services and APIs (e.g., embedding generation, translation, and LLM services), it cannot be executed or replicated outside of Foundry without significant modification.

## Acknowledgments

This project was completed under the mentorship and guidance of the Palantir Winter Defense Tech Fellowship team. Special thanks to the engineers and fellows who provided feedback throughout the development process.

