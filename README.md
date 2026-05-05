# CreditSenseAI: Intelligent Lending Decision Support System

## Project Overview

CreditSenseAI is an advanced credit analytics and automated lending decision system. It integrates traditional machine learning methodologies with modern agentic AI workflows to evaluate borrower risk profiles, predict default probabilities, and generate structured financial reasoning.

The application utilizes a multi-layered architecture to process numerical risk metrics while incorporating qualitative regulatory analysis via Retrieval-Augmented Generation (RAG).

## Core Architecture

The system is organized into the following subsystems:

1.  **Machine Learning Pipeline:** Utilizes Scikit-Learn to train and serve Logistic Regression and Decision Tree models based on historical borrower data.
2.  **Agentic Workflow:** Employs LangGraph to orchestrate a state machine where a Large Language Model (Groq Llama 3) evaluates the ML outputs against borrower context.
3.  **Regulatory Retrieval:** Implements FAISS vector search to query policy documents, ensuring lending decisions comply with established guidelines.
4.  **Bias & Fairness Detection:** Analyzes model output across demographic segments to calculate the Disparate Impact Ratio, flagging potential algorithmic bias.
5.  **Market Data Integration:** Connects to the US Treasury Fiscal Data API to fetch real-time interest rates for benchmark comparisons.

## Technology Stack

-   **Frontend:** Streamlit
-   **Machine Learning:** Scikit-Learn, Pandas, NumPy
-   **Agent Orchestration:** LangGraph
-   **Language Models:** Groq API
-   **Vector Storage:** FAISS, Sentence-Transformers
-   **Data Acquisition:** Requests (Treasury API, Google News RSS)

## Installation and Setup

### Prerequisites
-   Python 3.10 or higher
-   A valid Groq API Key

### Configuration

1.  Clone the repository and navigate to the project root.
2.  Initialize a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure environment variables by creating a `.env` file in the root directory:
    ```text
    GROQ_API_KEY=your_api_key_here
    ```

## Usage

To launch the Streamlit dashboard, execute the following command:

```bash
streamlit run app.py
```

The application will be accessible via a web browser at `http://localhost:8501`. 

### Available Modules
-   **Risk Assessment:** Input borrower parameters to receive a default probability, LLM-generated rationale, and confidence score.
-   **Bias Analysis:** Evaluate the underlying ML models for disparate impact across age, income, and homeownership status.
-   **Market Rates:** View current Treasury benchmark rates and compare them against specific borrower terms.
-   **Model Evaluation:** Review quantitative metrics (ROC-AUC, F1-Score) and feature importance charts for the active ML pipelines.
-   **AI Chat Interface:** Query the system regarding credit risk principles, augmented by live context fetched from financial news feeds.
