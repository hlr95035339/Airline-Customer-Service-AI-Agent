# Airline Customer Service AI Agent
---

## Project Overview
This project upgrades the classic *Airline Passenger Satisfaction Analysis* into a full AI Agent application.  
It combines RAG (Retrieval-Augmented Generation) with Agent workflows to build an intelligent airline customer service system.  
The agent can:
- Analyze customer sentiment based on historical satisfaction data
- Retrieve airline policies (refunds, delays, FAQs)
- Generate empathetic, high-EQ complaint response letters 
---

## Key Features
-  Sentiment Analysis: Detects positive, negative, or neutral emotions in customer feedback
-  RAG Knowledge Retrieval: Searches airline FAQs, refund rules, and delay compensation policies
-  High-EQ Reply Generation: Produces professional, empathetic responses in English or Chinese
-  Frontend UI: Interactive interface for customer input and AI replies
-  Language Detection: Automatically switches between English and Chinese replies

---

## System Architecture
```mermaid
graph TD
A[Frontend UI] --> B[FastAPI /chat Endpoint]
B --> C[Sentiment Analysis]
B --> D[RAG Knowledge Retrieval]
B --> E[High EQ Reply Generator]
D --> F[Policy Documents (FAISS Index)]
