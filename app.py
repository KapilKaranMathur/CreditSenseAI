"""
Credit Risk AI — Streamlit Application
========================================

Unified interface for credit risk analysis with 4 tabs:
1. Risk Assessment — ML scoring + agentic workflow + confidence breakdown + similar borrowers
2. Bias & Fairness — demographic bias detection across age, income, home ownership
3. Market Rates — live US Treasury rates with borrower rate comparison
4. Model Evaluation — full metrics dashboard with ROC curves, confusion matrices, feature importance
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.model_loader import predict_risk
from agent.schema import validate_input
from agent.exceptions import AgentWorkflowError
from agent.api_schema import AnalyzeRequest
from agent.service import AnalysisService



st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Monochrome CSS Override ──────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────── */
    .stApp { background-color: #0a0a0a; }
    h1, h2, h3, h4 { color: #f0f0f0 !important; letter-spacing: -0.02em; }
    p, li, span, label { color: #d4d4d4 !important; }

    /* ── Remove default colored alerts ──────────────────── */
    .stAlert > div { background-color: #1a1a1a !important; border: 1px solid #333 !important; color: #d4d4d4 !important; }
    .stAlert [data-testid="stMarkdownContainer"] p { color: #d4d4d4 !important; }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #888;
        border: none; border-bottom: 2px solid transparent;
        padding: 12px 24px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { color: #f0f0f0 !important; border-bottom: 2px solid #22c55e !important; }

    /* ── Metrics ────────────────────────────────────────── */
    [data-testid="stMetricValue"] { color: #f0f0f0 !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #888 !important; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }

    /* ── Dataframes ─────────────────────────────────────── */
    .stDataFrame { border: 1px solid #333; border-radius: 8px; }

    /* ── Buttons ────────────────────────────────────────── */
    .stButton > button {
        background-color: #1a1a1a; color: #f0f0f0; border: 1px solid #333;
        border-radius: 6px; font-weight: 500; transition: all 0.2s;
    }
    .stButton > button:hover { background-color: #22c55e; color: #0a0a0a; border-color: #22c55e; }
    .stButton > button[kind="primary"] { background-color: #22c55e !important; color: #0a0a0a !important; border: none; }

    /* ── Expanders ──────────────────────────────────────── */
    .streamlit-expanderHeader { background-color: #141414 !important; border: 1px solid #252525; border-radius: 6px; }

    /* ── Inputs ─────────────────────────────────────────── */
    .stTextInput > div > div { background-color: #141414; border: 1px solid #333; border-radius: 6px; }
    .stNumberInput > div > div { background-color: #141414; }
    .stSelectbox > div > div { background-color: #141414; }

    /* ── Section dividers ──────────────────────────────── */
    hr { border-color: #252525 !important; margin: 2rem 0 !important; }

    /* ── Sidebar ────────────────────────────────────────── */
    [data-testid="stSidebar"] { background-color: #0f0f0f; border-right: 1px solid #1f1f1f; }

    /* ── Progress bars (confidence) ─────────────────────── */
    .stProgress > div > div > div { background-color: #22c55e !important; }

    /* ── Chat messages ─────────────────────────────────── */
    [data-testid="stChatMessage"] { background-color: #141414; border: 1px solid #252525; border-radius: 8px; }

    /* ── Bar chart ──────────────────────────────────────── */
    .stBarChart { border: 1px solid #252525; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Credit Risk AI")
    st.caption("Multi-Agent Lending Decision Support System")
    st.markdown("---")
    st.markdown("**Components**")
    st.markdown("- 🧠 ML Risk Scoring")
    st.markdown("- 🤖 Agentic LLM Reasoning")
    st.markdown("- 📚 RAG Regulatory Retrieval")
    st.markdown("- 📊 Bias & Fairness Analysis")
    st.markdown("- 💹 Live Interest Rates")
    st.markdown("- 💬 AI Chat Assistant")
    st.markdown("---")

    st.subheader("🔑 API Configuration")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Required for LLM Reasoning and AI Chat. Get one free at console.groq.com"
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    elif not os.environ.get("GROQ_API_KEY"):
        st.caption("⚠️ Enter API key to enable LLM features.")

service = AnalysisService()


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════


def collect_borrower_input(prefix: str = "") -> dict:
    """Render the borrower input form and return feature dict."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        person_age = st.number_input(f"Age{prefix}", min_value=18, max_value=100, value=30)
        person_income = st.number_input(f"Annual Income ($){prefix}", min_value=0, value=50000, step=1000)
        person_emp_length = st.number_input(f"Employment Length (years){prefix}", min_value=0.0, value=5.0, step=0.5)
        person_home_ownership = st.selectbox(f"Home Ownership{prefix}", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col2:
        st.subheader("Loan Information")
        loan_amnt = st.number_input(f"Loan Amount ($){prefix}", min_value=1, value=10000, step=500)
        loan_int_rate = st.number_input(f"Interest Rate (%){prefix}", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        loan_intent = st.selectbox(
            f"Loan Intent{prefix}",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = st.selectbox(f"Loan Grade{prefix}", ["A", "B", "C", "D", "E", "F", "G"])

    st.subheader("Credit History")
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        cb_person_cred_hist_length = st.number_input(f"Credit History Length (years){prefix}", min_value=0, value=5)
    with ch_col2:
        cb_person_default_on_file = st.selectbox(f"Historical Default on File{prefix}", ["N", "Y"])

    loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0

    return {
        "person_age": int(person_age),
        "person_income": int(person_income),
        "person_emp_length": float(person_emp_length),
        "loan_amnt": int(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
    }




def render_confidence_breakdown(decision: dict, risk: dict) -> None:
    """Render the confidence score breakdown with visual gauges."""
    from agent.confidence_display import compute_confidence_breakdown

    breakdown = compute_confidence_breakdown(decision, risk)

    st.markdown("---")
    st.subheader("🎯 Confidence Score Breakdown")

    # Overall score
    score = breakdown["overall_score"]
    label = breakdown["overall_label"]

    score_col, label_col = st.columns([3, 1])
    with score_col:
        st.progress(score, text=f"Overall Confidence: {score:.0%}")
    with label_col:
        st.markdown(f"**{label} Confidence**")

    # Individual signals
    for signal in breakdown["signals"]:
        sig_col1, sig_col2, sig_col3 = st.columns([2, 1, 3])
        with sig_col1:
            st.caption(signal["name"])
        with sig_col2:
            st.caption(f"{signal['score']:.0%} × {signal['weight']:.0%}")
        with sig_col3:
            st.progress(signal["score"])
    st.caption(breakdown["signals"][0]["description"])
    st.caption(breakdown["signals"][1]["description"])
    st.caption(breakdown["signals"][2]["description"])


def render_similar_borrowers(input_data: dict) -> None:
    """Render the similar borrowers table."""
    from agent.similar_borrowers import find_similar_borrowers

    st.markdown("---")
    st.subheader("👥 Similar Historical Borrowers")

    try:
        matches = find_similar_borrowers(input_data, top_k=5)

        if not matches:
            st.markdown("No similar borrowers found.")
            return

        # Summary stats
        default_count = sum(1 for m in matches if m["loan_status"] == 1)
        st.markdown(
            f"Found **{len(matches)}** similar borrowers — "
            f"**{default_count}/{len(matches)}** defaulted "
            f"({default_count / len(matches):.0%} default rate)"
        )

        # Build display table
        display_cols = [
            "person_age", "person_income", "loan_amnt", "loan_int_rate",
            "loan_grade", "loan_intent", "outcome_label", "similarity_score",
        ]
        df = pd.DataFrame(matches)[display_cols]
        df.columns = [
            "Age", "Income", "Loan Amt", "Rate %",
            "Grade", "Intent", "Outcome", "Similarity",
        ]
        df["Income"] = df["Income"].apply(lambda x: f"${x:,.0f}")
        df["Loan Amt"] = df["Loan Amt"].apply(lambda x: f"${x:,.0f}")
        df["Rate %"] = df["Rate %"].apply(lambda x: f"{x:.1f}%")
        df["Similarity"] = df["Similarity"].apply(lambda x: f"{(1 - x):.0%}")

        st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.markdown(f"⚠️ Could not find similar borrowers: {e}")


def render_risk_assessment_tab() -> None:
    """Tab 1: Unified Risk Assessment."""
    st.header("🏦 Risk Assessment")
    st.caption("Submit a borrower profile for ML risk scoring, agentic LLM reasoning, confidence analysis, and historical borrower matching.")

    input_data = collect_borrower_input()

    analysis_mode = st.radio(
        "Analysis Mode",
        ["ML Scoring Only", "Full Agent Workflow (ML + LLM + RAG)"],
        horizontal=True,
    )

    model_name = st.radio(
        "ML Model",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True,
    )

    if st.button("🔍 Analyze Risk", type="primary"):
        backend_model = "logistic" if model_name == "Logistic Regression" else "decision_tree"

        try:
            # ── ML Scoring (always runs) ──────────────────────────────────
            validated_request = AnalyzeRequest(**input_data)
            validated_df = validate_input(validated_request.model_dump())
            prediction = predict_risk(validated_df, model_name=backend_model)

            st.markdown("---")
            st.subheader("📊 ML Risk Prediction")

            d_cols = st.columns(3)
            d_cols[0].metric("Predicted Class", prediction["label"])
            d_cols[1].metric("Default Probability", f"{prediction['probability']:.2%}")
            d_cols[2].metric("Model Used", prediction["model_used"].replace("_", " ").title())

            if prediction["label"] == "High Risk":
                st.markdown("⚠️ The ML model predicts **elevated default risk**.")
            else:
                st.markdown("✅ The ML model predicts **lower default risk**.")

            # ── Agentic Workflow (if selected) ────────────────────────────
            if analysis_mode == "Full Agent Workflow (ML + LLM + RAG)":
                st.markdown("---")
                st.subheader("🤖 Agentic Lending Decision")

                req = AnalyzeRequest(**input_data)
                with st.spinner("Running agent workflow (ML → Explain → RAG → LLM → Reflect)..."):
                    result = service.analyze(req)

                # Decision display
                decision = result.decision.lending_decision
                if decision == "REJECT":
                    st.markdown("🚫 **Decision: REJECT**")
                elif decision == "CONDITIONAL":
                    st.markdown("⏳ **Decision: CONDITIONAL**")
                else:
                    st.markdown("✅ **Decision: APPROVE**")

                info_cols = st.columns(3)
                info_cols[0].metric("Workflow Status", result.status.title())
                info_cols[1].metric("Reasoning Passes", str(result.reasoning_passes))
                info_cols[2].metric("Risk Tier", result.risk_tier or "N/A")

                report_cols = st.columns(2)
                with report_cols[0]:
                    st.write("**Borrower Profile Summary**")
                    st.write(result.decision.borrower_profile_summary)
                    st.write("**Risk Analysis**")
                    st.write(result.decision.risk_analysis)
                with report_cols[1]:
                    st.write("**Regulatory References**")
                    if result.decision.regulatory_references:
                        for source in result.decision.regulatory_references:
                            st.write(f"- {source}")
                    else:
                        st.write("- None")
                    st.write("**Disclaimer**")
                    st.write(result.decision.disclaimer)

                # Warnings
                if result.warnings:
                    with st.expander("⚠️ Warnings"):
                        for warning in result.warnings:
                            st.caption(f"{warning.code}: {warning.message}")

                # Confidence breakdown
                decision_dict = result.decision.model_dump(by_alias=True)
                risk_dict = result.risk.model_dump() if result.risk else prediction
                render_confidence_breakdown(decision_dict, risk_dict)

                # Workflow trace
                with st.expander("🔬 Workflow Trace"):
                    st.write(result.steps_completed)
                    st.json({
                        "risk_tier": result.risk_tier,
                        "confidence_score": result.confidence_score,
                        "metadata": result.metadata,
                    })

            # ── Similar Borrowers (always shown) ─────────────────────────
            render_similar_borrowers(input_data)

        except AgentWorkflowError as exc:
            st.markdown(f"⚠️ Request failed: {exc.message}")
        except Exception as exc:
            st.markdown(f"⚠️ Error: {exc}")




def _render_bias_group(title: str, metrics_df: pd.DataFrame, dir_result: dict) -> None:
    """Render a single bias analysis group with chart and DIR."""
    st.subheader(title)

    # DIR indicator
    if dir_result["is_fair"]:
        st.markdown(f"✅ {dir_result['assessment']}")
    else:
        st.markdown(f"⚠️ {dir_result['assessment']}")

    # Format the dataframe for display
    display_df = metrics_df.copy()
    display_df["Approval Rate"] = display_df["Approval Rate"].apply(lambda x: f"{x:.1%}")
    display_df["Avg Default Probability"] = display_df["Avg Default Probability"].apply(lambda x: f"{x:.1%}")
    display_df["Actual Default Rate"] = display_df["Actual Default Rate"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Bias Report as CSV",
        data=csv_data,
        file_name="bias_report.csv",
        mime="text/csv",
    )
    # Bar chart of approval rates
    chart_df = metrics_df[["Group", "Approval Rate"]].copy()
    chart_df = chart_df.set_index("Group")
    st.bar_chart(chart_df, color="#e5e5e5")


def render_bias_tab() -> None:
    """Tab 2: Bias & Fairness Analysis."""
    st.header("⚖️ Bias & Fairness Analysis")
    st.caption(
        "Analyze model predictions across demographic groups to detect potential bias. "
        "The Disparate Impact Ratio (DIR) measures fairness: DIR ≥ 0.80 is considered fair."
    )

    model_choice = st.radio(
        "Model to Analyze",
        ["Logistic Regression", "Decision Tree"],
        horizontal=True,
        key="bias_model_choice",
    )

    if st.button("🔍 Run Bias Analysis", type="primary", key="run_bias"):
        backend_model = "logistic" if model_choice == "Logistic Regression" else "decision_tree"

        with st.spinner("Running bias analysis across demographic groups..."):
            try:
                from agent.bias_detector import run_bias_analysis
                results = run_bias_analysis(model_name=backend_model)

                st.metric("Total Samples Analyzed", f"{results['total_samples']:,}")

                # Age analysis
                _render_bias_group(
                    "👤 Age Group Analysis",
                    results["age_analysis"],
                    results["age_dir"],
                )

                st.markdown("---")

                # Income analysis
                _render_bias_group(
                    "💰 Income Group Analysis",
                    results["income_analysis"],
                    results["income_dir"],
                )

                st.markdown("---")

                # Home ownership analysis
                _render_bias_group(
                    "🏠 Home Ownership Analysis",
                    results["ownership_analysis"],
                    results["ownership_dir"],
                )

            except Exception as e:
                st.markdown(f"⚠️ Bias analysis failed: {e}")



@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch_rates():
    """Fetch and cache Treasury rates for 1 hour."""
    from agent.rate_fetcher import fetch_treasury_rates
    return fetch_treasury_rates()


def render_market_rates_tab() -> None:
    """Tab 3: Real-Time Market Rates."""
    st.header("💹 Real-Time Market Interest Rates")
    st.caption("Current US Treasury average interest rates from the US Treasury Fiscal Data API (updated monthly).")

    with st.spinner("Fetching live Treasury rates..."):
        treasury_data = _cached_fetch_rates()

    if treasury_data.get("error"):
        st.markdown(f"⚠️ {treasury_data['error']}")
        return

    # Rates table
    st.subheader("📈 Current Treasury Rates")
    rates_df = pd.DataFrame(treasury_data["rates"])
    if not rates_df.empty:
        rates_df.columns = ["Security", "Rate (%)", "Type"]
        rates_df["Rate (%)"] = rates_df["Rate (%)"].apply(lambda x: f"{x:.3f}%")
        st.dataframe(rates_df, use_container_width=True, hide_index=True)

    # Borrower comparison
    st.markdown("---")
    st.subheader("🔍 Compare Your Borrower's Rate")

    borrower_rate = st.number_input(
        "Borrower's Interest Rate (%)",
        min_value=0.0, max_value=50.0, value=10.0, step=0.1,
        key="rate_comparison_input",
    )

    if st.button("Compare Rate", key="compare_rate"):
        from agent.rate_fetcher import compare_borrower_rate
        comparison = compare_borrower_rate(borrower_rate, treasury_data)

        if comparison["treasury_avg"] is not None:
            comp_cols = st.columns(3)
            comp_cols[0].metric("Borrower Rate", f"{comparison['borrower_rate']:.1f}%")
            comp_cols[1].metric("Treasury Average", f"{comparison['treasury_avg']:.3f}%")
            comp_cols[2].metric("Spread", f"+{comparison['spread']:.3f}%" if comparison["spread"] > 0 else f"{comparison['spread']:.3f}%")

            st.markdown(f"📊 {comparison['spread_assessment']}")

            # Per-security comparison
            with st.expander("Detailed Comparison by Security"):
                comp_df = pd.DataFrame(comparison["comparisons"])
                comp_df.columns = ["Security", "Treasury Rate (%)", "Spread (%)", "Direction"]
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("⚠️ Treasury rates not available for comparison.")



def render_model_evaluation_tab() -> None:
    """Tab 4: Full Model Evaluation Metrics Dashboard."""
    st.header("📊 Model Evaluation Dashboard")
    st.caption("Comprehensive metrics for both trained models with ROC curves, confusion matrices, and feature importance.")

    if st.button("🔄 Compute Full Metrics", type="primary", key="run_metrics"):
        with st.spinner("Computing comprehensive metrics for both models..."):
            try:
                from agent.metrics_dashboard import (
                    compute_comparison_metrics,
                    plot_roc_comparison,
                    plot_feature_importance,
                )

                comparison = compute_comparison_metrics()

                # ── Side-by-side summary ─────────────────────────────────
                st.subheader("📋 Model Comparison")
                summary_data = []
                for model_key, m in comparison.items():
                    label = "Logistic Regression" if model_key == "logistic" else "Decision Tree"
                    summary_data.append({
                        "Model": label,
                        "Accuracy": f"{m['accuracy']:.4f}",
                        "Precision": f"{m['precision']:.4f}",
                        "Recall": f"{m['recall']:.4f}",
                        "F1-Score": f"{m['f1']:.4f}",
                        "ROC-AUC": f"{m['roc_auc']:.4f}",
                        "Log Loss": f"{m['log_loss_val']:.4f}",
                        "MCC": f"{m['mcc']:.4f}",
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                csv_data = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Metrics Summary as CSV",
                    data=csv_data,
                    file_name="model_metrics_summary.csv",
                    mime="text/csv",
                )

                st.markdown("---")

                # ── ROC Curves ───────────────────────────────────────────
                st.subheader("📈 ROC Curve Comparison")
                roc_fig = plot_roc_comparison(comparison)
                st.pyplot(roc_fig)

                st.markdown("---")

                # ── Per-model details ────────────────────────────────────
                for model_key, m in comparison.items():
                    label = "Logistic Regression" if model_key == "logistic" else "Decision Tree"
                    st.subheader(f"🔬 {label}")

                    detail_cols = st.columns(4)
                    detail_cols[0].metric("Accuracy", m["accuracy"])
                    detail_cols[1].metric("F1-Score", m["f1"])
                    detail_cols[2].metric("ROC-AUC", m["roc_auc"])
                    detail_cols[3].metric("MCC", m["mcc"])

                    # Confusion Matrix
                    cm = m["confusion_matrix"]
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual: No Default", "Actual: Default"],
                        columns=["Predicted: No Default", "Predicted: Default"],
                    )
                    st.write("**Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=True)

                    # Classification Report
                    with st.expander("Classification Report"):
                        report = m["classification_report"]
                        report_df = pd.DataFrame(report).T
                        st.dataframe(report_df.round(4), use_container_width=True)

                    # Feature Importance
                    st.write("**Feature Importance**")
                    importance_fig = plot_feature_importance(m)
                    st.pyplot(importance_fig)

                    st.markdown("---")

            except Exception as e:
                st.markdown(f"⚠️ Metrics computation failed: {e}")




def render_chat_tab() -> None:
    """Tab 5: AI Chat with article fetching."""
    st.header("💬 AI Credit Risk Chat")
    st.caption("Ask questions about credit risk, lending, regulations, or financial analysis. The AI can also fetch relevant news articles from the web.")

    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "fetched_articles" not in st.session_state:
        st.session_state.fetched_articles = []

    # Article search sidebar
    with st.expander("🔍 Fetch Financial Articles", expanded=False):
        article_query = st.text_input(
            "Search for articles",
            placeholder="e.g., credit risk management 2024, RBI lending guidelines",
            key="article_query",
        )
        if st.button("Fetch Articles", key="fetch_articles"):
            if article_query:
                from agent.chat import fetch_financial_articles
                with st.spinner("Fetching articles..."):
                    articles = fetch_financial_articles(article_query, max_articles=5)
                st.session_state.fetched_articles = articles

                if articles:
                    st.markdown(f"✅ Found **{len(articles)}** articles")
                else:
                    st.markdown("⚠️ No articles found. Try a different query.")

        # Display fetched articles
        if st.session_state.fetched_articles:
            st.markdown("**📰 Fetched Articles** (will be used as context for your chat):")
            for i, article in enumerate(st.session_state.fetched_articles, 1):
                st.markdown(f"{i}. [{article['title']}]({article['link']}) — *{article['source']}*")

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.fetched_articles = []
            st.rerun()

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about credit risk, lending, regulations...")

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        from agent.chat import get_chat_response

        # Build conversation history (last 10 messages for context window)
        history = st.session_state.chat_messages[-10:-1] if len(st.session_state.chat_messages) > 1 else []

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(
                    user_message=user_input,
                    articles=st.session_state.fetched_articles or None,
                    conversation_history=history,
                )
            st.markdown(response)

        # Save assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

tab_risk, tab_bias, tab_rates, tab_eval, tab_chat = st.tabs([
    "🏦 Risk Assessment",
    "⚖️ Bias & Fairness",
    "💹 Market Rates",
    "📊 Model Evaluation",
    "💬 AI Chat",
])

with tab_risk:
    render_risk_assessment_tab()

with tab_bias:
    render_bias_tab()

with tab_rates:
    render_market_rates_tab()

with tab_eval:
    render_model_evaluation_tab()

with tab_chat:
    render_chat_tab()

