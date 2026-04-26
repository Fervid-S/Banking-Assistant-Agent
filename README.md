# Banking FAQ Assistant

A production-grade Agentic AI Capstone project built with LangGraph, ChromaDB, and Streamlit. This Banking FAQ Assistant handles customer queries about account types, loans, interest rates, fund transfer charges, and more — powered by a retrieval-augmented generation (RAG) pipeline with faithfulness evaluation.

---

## Project Structure

```
banking-faq-assistant/
├── bank_bot/
│   ├── __init__.py
│   ├── state.py
│   ├── config.py
│   ├── tools.py
│   ├── nodes.py
│   └── graph.py
├── knowledge_base/
│   ├── savings_account_types.txt
│   ├── fixed_deposit_rates.txt
│   ├── rtgs_neft_imps_charges.txt
│   ├── personal_loan_features.txt
│   ├── home_loan_eligibility.txt
│   ├── gold_loan_eligibility.txt
│   ├── credit_card_variants.txt
│   ├── current_account_variants.txt
│   ├── atm_debit_card_charges.txt
│   ├── locker_facility_charges.txt
│   ├── kyc_account_opening.txt
│   └── insurance_products.txt
├── tests/
│   └── ragas_eval.py
├── capstone_streamlit.py
├── ingest.py
├── requirements.txt
├── .env
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites

- Python **3.10** or higher
- A [Groq API Key](https://console.groq.com/)

---

### 2. Installation

Clone the repository and set up the virtual environment:

```bash
git clone https://github.com/your-username/banking-faq-assistant.git
cd banking-faq-assistant
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

### 3. Environment Variables

Create a `.env` file in the root directory:

```plaintext
GROQ_API_KEY=your_actual_api_key_here
```

---

### 4. Knowledge Base Setup

Ensure your 12 banking documents are placed inside the `/knowledge_base` folder. These files cover:

- Savings Account Types (Regular, Premium, Senior, Salary, Women's, Minor)
- Loan Products (Home Loan, Personal Loan, Gold Loan)
- Fixed Deposit Rates and Charges
- Fund Transfer Services (RTGS, NEFT, IMPS, UPI)
- Credit Card Variants (Classic, Signature, Infinite)
- Current Account Variants (Basic, Business, Premium, Trade)
- ATM & Debit Card Charges and Limits
- Locker Facility Charges and Allotment Policy
- KYC Requirements and Account Opening Procedures
- Insurance Products (Life, Health, Motor, Home)

---

### 5. Ingest the Knowledge Base

Before running the app for the first time, populate the ChromaDB vector store:

```bash
python ingest.py
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run capstone_streamlit.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Example Test Queries

| Category | Query | Expected Behaviour |
|---|---|---|
| **Persistence** | "I live in a rural area." → "What is my MAB?" | Returns ₹2,000 (rural MAB for Regular Savings) |
| **Logic** | "What is the interest rate for a self-employed architect with a CIBIL score of 780 for a personal loan?" | Returns 13.00% (750–799 band + 0.50% self-employed premium) |
| **Tool Use** | "Calculate EMI for ₹10 lakh at 9.5% for 5 years" | Runs loan_calculator tool and returns monthly EMI |
| **Date/Time** | "What is today's date?" | Invokes get_current_datetime tool |
| **Safety / Out-of-Scope** | "How do I book a movie ticket?" | Returns helpline fallback (1800-BANK-HELP) |
| **Greeting** | "Hello!" | Friendly greeting without KB lookup |

---

## Agent Flow

The LangGraph `StateGraph` routes every query through exactly **8 nodes**:

```
memory_node → router_node → retrieval_node → answer_node → eval_node → save_node
                          ↘ tool_node ↗
                          ↘ skip_node ↗
```

| Node | Responsibility |
|---|---|
| `memory_node` | Sliding-window history + customer name extraction |
| `router_node` | Routes to `retrieve`, `tool`, or `skip` |
| `retrieval_node` | ChromaDB top-K semantic search |
| `skip_node` | Handles greetings and trivial inputs |
| `tool_node` | Runs `loan_calculator` or `get_current_datetime` |
| `answer_node` | Generates grounded answer from retrieved context |
| `eval_node` | Faithfulness check with retry logic (max 2 retries) |
| `save_node` | Terminal node; extensible for audit logging |

---

## RAGAS Evaluation

Run the baseline evaluation against 10 curated test cases:

```bash
# Full RAGAS evaluation (requires OpenAI API key and embeddings)
python tests/ragas_eval.py

# Quick smoke test (no embeddings cost, ideal for CI)
python tests/ragas_eval.py --smoke
```

**Metrics evaluated:**

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question asked? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Does the retrieved context cover the ground truth? |

Results are saved to `tests/ragas_results.json` after each run.

---

## Design Principles

- **Strict grounding** — The `answer_node` system prompt forbids introducing facts absent from the retrieved context.
- **Faithfulness loop** — `eval_node` re-runs `answer_node` up to `MAX_EVAL_RETRIES` times if hallucination is detected.
- **Resilient tools** — `loan_calculator` and `get_current_datetime` never raise exceptions; they return error strings.
- **Lazy singletons** — LLM client and ChromaDB collection are initialised once via `@lru_cache` and `@st.cache_resource`.
- **Sliding memory** — Only the most recent `MEMORY_WINDOW` (default: 6) turn-pairs are sent to the LLM to manage token cost.

---

## License

This project is developed for educational purposes as part of a **Senior AI Engineering Capstone**. Not intended for production banking use.

