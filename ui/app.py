import os
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure page metadata
st.set_page_config(
    page_title="Customer Support Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Dark elegant styling */
    .reportview-container {
        background: #0f1115;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f1f3f5 !important;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .metric-card {
        background-color: #1e222b;
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ab0c5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .ticket-container {
        background-color: #161a22;
        border-left: 5px solid #4f46e5;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
        color: white;
    }
    .tag-high { background-color: #ef4444; }
    .tag-medium { background-color: #f59e0b; }
    .tag-low { background-color: #10b981; }
    .tag-pos { background-color: #10b981; }
    .tag-neg { background-color: #ef4444; }
    .tag-neu { background-color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# Read API URL from environment (set on Streamlit Cloud) or fall back to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Predefined ticket samples for quick demo
TICKET_SAMPLES = {
    "Select a Preset Sample...": "",
    "Billing/Refund Query": "Customer: Hi, I'm writing because I noticed a double charge of $29.00 on my invoice #9812-B. This happened yesterday on 2026-06-29. My email is customer.care@gmail.com and my credit card ends in 4111. I want a refund of the duplicate charge immediately.",
    "Account Reset Failure": "Customer: Help! I tried resetting my login password using the standard portal, but when I click the link from security@myauth.com, it gives me a blank page. The page URL is auth.platform.com/verify?code=98231. Phone: +1-555-019-2834. I'm locked out.",
    "System Slowdown Report": "Customer: The analytics page is incredibly slow. It took me 45 seconds to generate the monthly report. This is critical for our operations. Let me know if there's an outage on the West Coast clusters.",
    "Custom Text (Type below)": "Type your transcript here..."
}

# Session State initializations
if "token" not in st.session_state:
    st.session_state.token = ""
if "user_role" not in st.session_state:
    st.session_state.user_role = ""
if "username" not in st.session_state:
    st.session_state.username = ""
if "api_response" not in st.session_state:
    st.session_state.api_response = None
if "current_ticket_id" not in st.session_state:
    st.session_state.current_ticket_id = None

# --- SIDEBAR: LOGIN & CONNECTIONS ---
st.sidebar.title("🔐 Authentication")
if not st.session_state.token:
    username = st.sidebar.text_input("Username", value="agent_john")
    password = st.sidebar.text_input("Password", type="password", value="password123")
    if st.sidebar.button("Login"):
        try:
            res = requests.post(f"{API_URL}/api/token", data={"username": username, "password": password})
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                # Simple extraction of role
                st.session_state.user_role = "admin" if "admin" in username else "agent"
                st.sidebar.success(f"Logged in as {username} ({st.session_state.user_role})")
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials.")
        except Exception as e:
            st.sidebar.error(f"Cannot connect to API: {e}")
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.username}`")
    st.sidebar.markdown(f"**Role:** `{st.session_state.user_role.upper()}`")
    if st.sidebar.button("Logout"):
        st.session_state.token = ""
        st.session_state.user_role = ""
        st.session_state.username = ""
        st.session_state.api_response = None
        st.session_state.current_ticket_id = None
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.title("ℹ️ Platform Info")
st.sidebar.info(
    "This workspace serves support agents by parsing conversations, "
    "redacting PII data, executing semantic RAG searches for past issues, "
    "and displaying agent runbooks."
)

# --- MAIN APP LAYOUT ---
st.title("🤖 Customer Support Intelligence Platform")
st.markdown("RAG-driven Ticket Summarisation, Intent Extraction, and Agent Recommendation")

if not st.session_state.token:
    st.warning("⚠️ Please log in from the sidebar to access the platform services.")
else:
    tabs = st.tabs(["📋 Agent Workspace", "📊 Supervisor Analytics", "🚀 Batch Ingestion"])

    # ==================== TAB 1: AGENT WORKSPACE ====================
    with tabs[0]:
        st.subheader("Ticket Ingestion & Parsing Sandbox")
        
        # Ingestion inputs
        col1, col2 = st.columns([1, 2])
        with col1:
            sample_choice = st.selectbox("Load Sample Ticket", list(TICKET_SAMPLES.keys()))
            default_text = TICKET_SAMPLES[sample_choice]
            if sample_choice == "Custom Text (Type below)":
                default_text = ""
        with col2:
            ticket_input = st.text_area("Support Transcript / Email Content", value=default_text, height=130)

        # Trigger Processing
        if st.button("Process & Analyze Ticket"):
            if not ticket_input.strip() or ticket_input == "Type your transcript here...":
                st.error("Please enter a valid ticket transcript.")
            else:
                with st.spinner("Processing RAG-driven generation pipelines..."):
                    # 1. Ingest/Index first
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    ingest_res = requests.post(f"{API_URL}/api/ingest", json={"text": ticket_input}, headers=headers)
                    
                    if ingest_res.status_code == 200:
                        ingest_data = ingest_res.json()
                        st.session_state.current_ticket_id = ingest_data["doc_id"]
                        
                        # 2. Query/Generate pipeline details
                        query_res = requests.post(
                            f"{API_URL}/api/query", 
                            json={"ticket_id": st.session_state.current_ticket_id}, 
                            headers=headers
                        )
                        if query_res.status_code == 200:
                            st.session_state.api_response = query_res.json()
                            st.success("Analysis complete!")
                        else:
                            st.error(f"Query Pipeline Error: {query_res.text}")
                    else:
                        st.error(f"Ingestion Pipeline Error: {ingest_res.text}")

        # Render Response Data if present
        if st.session_state.api_response:
            res_data = st.session_state.api_response
            
            st.markdown("---")
            c_left, c_right = st.columns([1, 1])

            # Left side: Preprocessing & PII Masking
            with c_left:
                st.markdown("### 🔒 Conversation & PII Masking")
                st.markdown("**Processed / Masked Text:**")
                st.info(res_data.get("pii_masked_text", ""))
                
                # Metadata / Entity Tags
                st.markdown("### 🏷️ Metadata & Intent Classification")
                meta = res_data.get("metadata", {})
                
                # SLA Priority & Sentiment tags
                sla_val = meta.get("sla_priority", "Low")
                sent_val = meta.get("sentiment", "Neutral")
                
                st.markdown(
                    f"**SLA Priority:** <span class='tag tag-{sla_val.lower()}'>{sla_val}</span> "
                    f"**Sentiment:** <span class='tag tag-{sent_val.lower()[:3]}'>{sent_val}</span>", 
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**Primary Intent:** `{meta.get('intent', 'General')}`")
                st.markdown(f"**Products Mentioned:** `{', '.join(meta.get('products', [])) or 'None'}`")
                
                # Named Entities
                entities = meta.get("entities", [])
                if entities:
                    st.markdown("**Extracted Named Entities:**")
                    ent_df = pd.DataFrame(entities)
                    st.table(ent_df)
                else:
                    st.markdown("**Extracted Named Entities:** None found.")

            # Right side: Summary & Recommendations
            with c_right:
                st.markdown("### 💡 AI Generated Ticket Summary")
                st.write(res_data.get("summary", ""))

                st.markdown("### 🎯 Recommended Next-Actions")
                recs = res_data.get("recommendations", {})
                
                for act in recs.get("recommended_actions", []):
                    st.markdown(f"- ✅ {act}")
                    
                # Canned articles
                st.markdown("#### 📚 Relevant Knowledge Base Articles")
                kb_list = recs.get("kb_articles", [])
                for kb in kb_list:
                    st.markdown(
                        f"- [{kb.get('title')}]({kb.get('url')}) "
                        f"*(Confidence: `{int(kb.get('confidence', 0)*100)}%`)*"
                    )

                # Agent Runbook
                runbook = recs.get("runbook", {})
                st.markdown(f"#### 🪜 Step-by-Step Troubleshooting Runbook (Estimated Time: `{runbook.get('estimated_time_mins', 5)} mins`)")
                for i, step in enumerate(runbook.get("steps", [])):
                    st.markdown(f"{i+1}. {step}")

            # RAG Results Section
            st.markdown("### 🔍 Historical RAG Context (Top Vector DB Matches)")
            matches = res_data.get("retrieved_tickets", [])
            if matches:
                for idx, match in enumerate(matches):
                    with st.expander(f"Similar Ticket #{idx+1} (Vector Score: {match.get('score', 0):.4f})"):
                        st.markdown(f"**ID:** `{match.get('id')}`")
                        st.markdown(f"**Past Intent:** `{match.get('metadata', {}).get('intent', 'General')}`")
                        st.text(match.get("text", ""))
            else:
                st.write("No historical tickets found.")

            # Feedback Loop Interface
            st.markdown("---")
            st.markdown("### ✍️ Human-in-the-Loop Feedback Portal")
            
            with st.form("feedback_form"):
                st.markdown("Rate the accuracy and usefulness of the AI outputs:")
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    r_sum = st.slider("Summary Quality", 1, 5, 5)
                with f_col2:
                    r_int = st.slider("Intent Extraction", 1, 5, 5)
                with f_col3:
                    r_rec = st.slider("Action Relevance", 1, 5, 5)

                st.markdown("Correct / Override AI generation if needed:")
                corr_sum = st.text_area("Corrected Summary (leave blank to accept AI summary)", value="")
                corr_int = st.text_input("Corrected Intent (leave blank to accept AI intent)", value="")
                comments = st.text_input("Additional Comments")

                submit_fb = st.form_submit_button("Submit Agent Review & Log Feedback")
                if submit_fb:
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    feedback_body = {
                        "ticket_id": st.session_state.current_ticket_id,
                        "query_id": res_data.get("query_id"),
                        "summary_rating": r_sum,
                        "intent_rating": r_int,
                        "recommendation_rating": r_rec,
                        "comments": comments,
                        "corrected_summary": corr_sum if corr_sum.strip() else None,
                        "corrected_intent": corr_int if corr_int.strip() else None
                    }
                    fb_res = requests.post(f"{API_URL}/api/feedback", json=feedback_body, headers=headers)
                    if fb_res.status_code == 200:
                        st.success("Feedback successfully logged in SQL Database! Future RAG queries will leverage corrected text.")
                    else:
                        st.error(f"Feedback Log Failure: {fb_res.text}")

    # ==================== TAB 2: SUPERVISOR ANALYTICS ====================
    with tabs[1]:
        st.subheader("Supervisor Observability & MLOps Performance")
        
        if st.session_state.user_role != "admin":
            st.warning("🔒 This panel requires Admin privileges. Please log in as `admin_boss` to unlock analytics.")
        else:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                metrics_res = requests.get(f"{API_URL}/api/metrics", headers=headers)
                if metrics_res.status_code == 200:
                    metrics_data = metrics_res.json()

                    # High level metric cards
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-value'>{metrics_data.get('total_tickets')}</div>"
                            f"<div class='metric-label'>Tickets Ingested</div></div>", 
                            unsafe_allow_html=True
                        )
                    with m_col2:
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-value'>{metrics_data.get('total_feedbacks')}</div>"
                            f"<div class='metric-label'>Feedbacks Logged</div></div>", 
                            unsafe_allow_html=True
                        )
                    with m_col3:
                        avg_ratings = metrics_data.get("average_ratings", {})
                        total_avg = sum(avg_ratings.values()) / 3 if avg_ratings else 0.0
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-value'>{total_avg:.2f}/5</div>"
                            f"<div class='metric-label'>Average CSAT Rating</div></div>", 
                            unsafe_allow_html=True
                        )
                    with m_col4:
                        kappa = metrics_data.get("inter_rater_agreement", {}).get("cohens_kappa", 1.0)
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-value'>{kappa:.2f}</div>"
                            f"<div class='metric-label'>Cohen's Kappa (Agreement)</div></div>", 
                            unsafe_allow_html=True
                        )

                    st.markdown("---")
                    
                    # Intent Distribution Charts
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        st.markdown("### 📊 Distribution of Customer Intents")
                        intent_dist = metrics_data.get("intent_distribution", {})
                        if intent_dist:
                            df_intent = pd.DataFrame(list(intent_dist.items()), columns=["Intent", "Count"])
                            fig = px.bar(df_intent, x="Intent", y="Count", color="Count",
                                         color_continuous_scale="Viridis", template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No intent logs available yet.")

                    with g_col2:
                        st.markdown("### 🏆 AI Accuracy Ratings Breakdown")
                        ratings = metrics_data.get("average_ratings", {})
                        if ratings:
                            df_ratings = pd.DataFrame(list(ratings.items()), columns=["Component", "Rating"])
                            fig = px.line(df_ratings, x="Component", y="Rating", markers=True, template="plotly_dark")
                            fig.update_layout(yaxis=dict(range=[1, 5.2]))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No rating data available yet.")
                else:
                    st.error("Failed to load platform metrics.")
            except Exception as e:
                st.error(f"Metrics fetch error: {e}")

    # ==================== TAB 3: BATCH INGESTION ====================
    with tabs[2]:
        st.subheader("Data Upload Pipeline")
        st.write("Upload a CSV file containing columns `text` to ingest multiple tickets at once.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns:
                    st.error("CSV file must contain a column named `text`.")
                else:
                    st.write(df.head())
                    if st.button("Process Batch Upload"):
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        success_count = 0
                        
                        progress_bar = st.progress(0)
                        for idx, row in df.iterrows():
                            text_val = str(row["text"])
                            payload = {"text": text_val}
                            res = requests.post(f"{API_URL}/api/ingest", json=payload, headers=headers)
                            if res.status_code == 200:
                                success_count += 1
                            progress_bar.progress((idx + 1) / len(df))
                            
                        st.success(f"Batch completed: Ingested {success_count} out of {len(df)} tickets successfully.")
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")
