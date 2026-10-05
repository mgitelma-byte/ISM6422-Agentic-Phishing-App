import asyncio
import json
import os
import threading

import pandas as pd
import streamlit as st
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Phishing Triage",
    page_icon="🛡️",
    layout="wide",
)

set_tracing_disabled(True)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
      .small-note {color: #64748b; font-size: 0.9rem;}
      .status-card {
        border: 1px solid #d8e2ef;
        border-radius: 12px;
        padding: 14px 16px;
        background: #f8fbff;
        min-height: 100px;
      }
      .status-title {font-size: 0.82rem; color: #5b6b7a; margin-bottom: 5px;}
      .status-value {font-size: 1.35rem; font-weight: 700; color: #17324d;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Synthetic classroom dataset + classifier
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_classifier():
    legitimate_subjects = [
        "Project meeting reminder",
        "Updated benefits information",
        "Weekly team status",
        "Training registration confirmation",
        "Office maintenance notice",
        "Travel reimbursement update",
        "Quarterly planning meeting",
        "Library account notice",
        "Course registration reminder",
        "Vendor invoice received",
    ]

    legitimate_bodies = [
        "Hi team, this is a reminder that our project meeting is tomorrow at 10 AM. Please send agenda items before 4 PM.",
        "Human Resources has posted the updated benefits guide on the employee portal. Please review it when convenient.",
        "Here is the weekly status summary for our current projects. No action is required unless your item needs correction.",
        "Your registration for the internal training session is confirmed. The session will be held in Building A, Room 210.",
        "Facilities will perform scheduled maintenance this weekend. Some offices may have limited access Saturday morning.",
        "Your travel reimbursement was reviewed and approved. Payment will be included in the next normal reimbursement cycle.",
        "The quarterly planning meeting is scheduled for Friday. Please bring your department's current priorities and estimates.",
        "Your library loan is due next week. You may renew eligible items through the official library portal.",
        "Registration for the upcoming term opens Monday. Students should use the normal university registration system.",
        "Accounting received the vendor invoice and routed it for standard review. No immediate action is required from you.",
    ]

    phishing_subjects = [
        "URGENT: Account will be suspended",
        "Password expires today",
        "Payroll problem - verify now",
        "Unusual login detected",
        "Immediate mailbox verification required",
        "Refund waiting - confirm banking details",
        "Security alert - action required",
        "Invoice overdue - open attachment",
        "Final warning: storage quota exceeded",
        "Benefits account locked",
    ]

    phishing_bodies = [
        "Your account will be suspended today unless you verify your username and password immediately at http://account-check.invalid.",
        "Your password expires in 30 minutes. Click http://password-reset.invalid and enter your current credentials to keep access.",
        "Payroll could not process your deposit. Confirm your employee ID, password, and bank information at http://payroll-fix.invalid.",
        "We detected an unusual login. To prevent account closure, sign in immediately at http://security-login.invalid.",
        "Your mailbox has exceeded the limit. Verify your email password now at http://mail-quota.invalid to avoid deletion.",
        "A refund is waiting for you. Confirm your routing number and online banking credentials at http://refund-center.invalid.",
        "Security requires immediate verification. Enter your username, password, and MFA code at http://verify-now.invalid.",
        "The attached invoice is overdue. Open the file and enable content immediately to avoid a late fee.",
        "Final warning: your storage account will be deleted today. Sign in at http://storage-update.invalid to keep your files.",
        "Your benefits account has been locked. Restore access by confirming your SSN and password at http://benefits-help.invalid.",
    ]

    rows = []
    for i in range(6):
        for subject, body in zip(legitimate_subjects, legitimate_bodies):
            rows.append(
                {
                    "Email Text": f"Subject: {subject}\n\n{body} Reference {i + 1}.",
                    "Email Type": "Legitimate",
                }
            )
        for subject, body in zip(phishing_subjects, phishing_bodies):
            rows.append(
                {
                    "Email Text": f"Subject: {subject}\n\n{body} Case {i + 1}.",
                    "Email Type": "Phishing",
                }
            )

    rows.extend(
        [
            {
                "Email Text": "Subject: Final reminder\n\nRegistration closes tonight. Use the link in our official newsletter if you still want the discounted rate.",
                "Email Type": "Legitimate",
            },
            {
                "Email Text": "Subject: Benefits deadline\n\nPlease review your benefits selections by Friday using the normal employee portal. Contact HR with questions.",
                "Email Type": "Legitimate",
            },
            {
                "Email Text": "Subject: Updated security document\n\nPlease review the attached policy update before tomorrow's meeting. No password or login information is requested.",
                "Email Type": "Legitimate",
            },
            {
                "Email Text": "Subject: Executive request\n\nI am in a meeting and need you to purchase gift cards immediately. Reply with the card numbers and do not call me.",
                "Email Type": "Phishing",
            },
            {
                "Email Text": "Subject: Shared document\n\nA confidential document has been shared with you. Sign in at http://document-login.invalid with your email password to view it.",
                "Email Type": "Phishing",
            },
            {
                "Email Text": "Subject: Payment confirmation\n\nWe need to reprocess your payment. Send your banking password and verification code by reply email.",
                "Email Type": "Phishing",
            },
        ]
    )

    df = pd.DataFrame(rows)
    work = df[["Email Text", "Email Type"]].dropna().copy()
    work["target"] = work["Email Type"].map({"Legitimate": 0, "Phishing": 1})

    X_train, _, y_train, _ = train_test_split(
        work["Email Text"],
        work["target"],
        test_size=0.20,
        random_state=42,
        stratify=work["target"],
    )

    classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)
    return classifier


@st.cache_resource(show_spinner=False)
def get_agent_event_loop():
    """Keep one asyncio event loop alive for the lifetime of the Streamlit app."""
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return loop


def run_async(coroutine):
    """Run all Agents SDK async work on the same persistent event loop."""
    loop = get_agent_event_loop()
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    return future.result()


def build_agent_system(api_key, phishing_model):
    """Create the same two-agent / three-tool architecture used in the project."""

    agnes_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://apihub.agnes-ai.com/v1",
    )

    model = OpenAIChatCompletionsModel(
        model="agnes-3.0-flash",
        openai_client=agnes_client,
    )

    @function_tool
    def classify_email(email_text: str) -> dict:
        """Classify one email as Phishing or Legitimate using the trained ML model."""
        if not email_text or not email_text.strip():
            return {"error": "Email text is empty."}

        probability = float(phishing_model.predict_proba([email_text])[0, 1])
        prediction = int(probability >= 0.50)

        if probability >= 0.80:
            risk_level = "High"
        elif probability >= 0.50:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "classification": "Phishing" if prediction == 1 else "Legitimate",
            "phishing_probability": round(probability, 4),
            "risk_level": risk_level,
            "threshold": 0.50,
        }

    security_review_agent = Agent(
        name="Security Review Specialist",
        instructions=(
            "You are a cybersecurity review specialist. Review the email content and the ML classifier result "
            "provided by the Phishing Triage Assistant. Distinguish the model prediction from your qualitative review. "
            "Look for indicators such as credential requests, urgent pressure, suspicious or mismatched links, "
            "requests for sensitive information, and unusual sender behavior. "
            "Do not claim certainty. If information is insufficient, say what is missing. "
            "Do not take or claim to take real email actions."
        ),
        model=model,
    )

    security_review_tool = security_review_agent.as_tool(
        tool_name="security_review",
        tool_description=(
            "Review an email and its ML phishing classification for suspicious indicators "
            "and provide a concise security recommendation."
        ),
    )

    @function_tool(needs_approval=True)
    def quarantine_email(email_id: str, reason: str) -> str:
        """Simulate quarantining an email after explicit human approval."""
        if not email_id.strip():
            return "Mock quarantine blocked: email_id is required."
        if not reason.strip():
            return "Mock quarantine blocked: a reason is required."
       return (
    f"SIMULATED quarantine completed for {email_id}. "
    f"This is a classroom demonstration only and does not affect a real email system. "
    f"Reason: {reason}"
)

    triage_agent = Agent(
        name="Phishing Triage Assistant",
        instructions=(
            "You are an email-security triage assistant. Help users review suspicious emails. "
            "A machine-learning classifier is the source of the phishing/legitimate prediction. "
            "Do not invent a model score or claim that an email was classified unless the classifier tool was used. "
            "Explain that automated classification is a first-pass triage aid, not a guarantee. "
            "Do not click links, send email, delete messages, or claim to take real-world email actions."
        ),
        model=model,
    )

    triage_agent.tools = [classify_email, security_review_tool, quarantine_email]

    triage_agent.instructions += (
        " When the user asks to analyze an email, always call classify_email before giving a phishing verdict. "
        "Call security_review after classification when the classifier result is Phishing, when risk is Medium or High, "
        "or when the user explicitly asks for a deeper/security review. "
        "If the user explicitly asks to quarantine or take action, classify first and obtain security_review before deciding whether to call quarantine_email. "
        "Call quarantine_email only when the user explicitly requested quarantine/action and the evidence supports quarantine. "
        "Do not quarantine an email that the evidence indicates is legitimate. "
        "The quarantine tool requires human approval. "
"If approval is rejected, do not claim the email was quarantined. "
"If approval is granted, clearly state that the quarantine is SIMULATED for a classroom demonstration "
"and does not affect or interact with any real email system. "
"Never claim that a real email was isolated, blocked, deleted, or prevented from delivery. "
        "Use the email ID supplied by the user; if none is supplied, use DEMO-001. "
        "Consolidate the tool results into one concise answer and clearly state the ML classification, probability, specialist recommendation when used, "
        "and whether an action is pending approval, completed, or not appropriate."
    )

    return {
        "triage_agent": triage_agent,
        "security_review_agent": security_review_agent,
    }


def parse_jsonish(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def extract_trace(result):
    """Create a presentation-friendly trace and recover classifier output."""
    events = []
    classifier_output = None
    last_tool = None

    for item in getattr(result, "new_items", []):
        agent = getattr(getattr(item, "agent", None), "name", "Unknown")
        item_type = getattr(item, "type", "unknown")
        raw_item = getattr(item, "raw_item", None)

        if item_type == "tool_call_item" and getattr(raw_item, "type", None) == "function_call":
            last_tool = getattr(raw_item, "name", "unknown_tool")
            arguments = getattr(raw_item, "arguments", "")
            events.append(
                {
                    "event": "Tool call",
                    "agent": agent,
                    "tool": last_tool,
                    "details": arguments,
                }
            )

        elif item_type == "tool_call_output_item":
            output = parse_jsonish(getattr(item, "output", ""))
            events.append(
                {
                    "event": "Tool result",
                    "agent": agent,
                    "tool": last_tool or "tool",
                    "details": output,
                }
            )
            if last_tool == "classify_email" and isinstance(output, dict):
                classifier_output = output

    return events, classifier_output


def reset_analysis_state():
    for key in [
        "final_output",
        "trace",
        "classifier_output",
        "pending_state",
        "pending_request",
        "pending_action",
        "approval_status",
    ]:
        st.session_state.pop(key, None)


def default_api_key():
    # Environment variable is convenient on HiperGator/local machines.
    if os.environ.get("AGNES_API_KEY"):
        return os.environ["AGNES_API_KEY"]

    # Optional Streamlit Cloud secret.
    try:
        return st.secrets.get("AGNES_API_KEY", "")
    except Exception:
        return ""


PHISHING_DEMO = """Subject: Urgent account verification required

Your mailbox will be disabled today unless you verify your password immediately.
Click the link below to confirm your account and avoid suspension:
http://example-login-security.invalid/verify"""

LEGITIMATE_DEMO = """Subject: Benefits deadline

Please review your benefits selections by Friday using the normal employee portal.
Contact HR with questions."""


# ---------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------
if "email_text" not in st.session_state:
    st.session_state.email_text = PHISHING_DEMO

if "email_id" not in st.session_state:
    st.session_state.email_id = "DEMO-001"

if "trace" not in st.session_state:
    st.session_state.trace = []


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🛡️ Agentic Phishing Email Triage")
st.caption(
    "Group 4 classroom prototype • ML classification + specialist agent + human-in-the-loop quarantine"
)

st.info(
    "This is a first-pass classroom triage application. The quarantine action is simulated and does not connect to a real email system.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------
# Sidebar initialization
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Application Setup")
    api_key = st.text_input(
        "AGNES API Key",
        value=default_api_key(),
        type="password",
        help="Used only to run the AGNES model for this session.",
        key="agnes_key_input",
    )

    if st.button("Initialize / Reset Agent System", use_container_width=True, type="primary"):
        if not api_key.strip():
            st.error("Enter an AGNES API key first.")
        else:
            with st.spinner("Training classifier and initializing agents..."):
                phishing_model = train_classifier()
                st.session_state.agent_system = build_agent_system(
                    api_key.strip(), phishing_model
                )
                reset_analysis_state()
            st.success("Agentic application ready.")

    if "agent_system" in st.session_state:
        st.success("● System ready")
        st.markdown(
            "**Architecture**\n"
            "- Phishing Triage Assistant\n"
            "- Security Review Specialist\n"
            "- `classify_email`\n"
            "- `security_review`\n"
            "- `quarantine_email` + HITL"
        )
    else:
        st.warning("Initialize the agent system before analyzing an email.")

    st.divider()
    st.caption("OpenAI Agents SDK • AGNES 3.0 Flash • TF-IDF + Logistic Regression")


# ---------------------------------------------------------------------
# Demo loader + input
# ---------------------------------------------------------------------
left, right = st.columns([1.75, 1])

with left:
    st.subheader("Email to Analyze")

    scenario = st.selectbox(
        "Example",
        [
            "Demo 1 — Clear phishing",
            "Demo 2 — Legitimate but urgent-looking",
            "Custom email",
        ],
    )

    load_col, clear_col = st.columns(2)
    with load_col:
        if st.button("Load Selected Example", use_container_width=True):
            if scenario.startswith("Demo 1"):
                st.session_state.email_id = "DEMO-001"
                st.session_state.email_text = PHISHING_DEMO
            elif scenario.startswith("Demo 2"):
                st.session_state.email_id = "DEMO-002"
                st.session_state.email_text = LEGITIMATE_DEMO
            reset_analysis_state()
            st.rerun()

    with clear_col:
        if st.button("Clear Results", use_container_width=True):
            reset_analysis_state()
            st.rerun()

    st.text_input("Email ID", key="email_id")
    st.text_area("Email content", height=250, key="email_text")

    action_requested = st.checkbox(
        "If dangerous, request quarantine",
        value=True,
        help="When checked, the agent may request the HITL quarantine tool if the evidence supports it.",
    )

    analyze_clicked = st.button(
        "Analyze Email",
        type="primary",
        use_container_width=True,
        disabled="agent_system" not in st.session_state,
    )

with right:
    st.subheader("Agentic Workflow")
    st.markdown(
        """
        **1. Triage Agent** interprets the request  
        ↓  
        **2. `classify_email`** produces ML evidence  
        ↓  
        **3. Security Specialist** reviews suspicious cases  
        ↓  
        **4. `quarantine_email`** is requested only when justified  
        ↓  
        **5. Human Approval** is required before the mock action
        """
    )
    st.caption(
        "The path is dynamic: low-risk legitimate messages can stop without quarantine."
    )


# ---------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------
if analyze_clicked:
    reset_analysis_state()

    email_text = st.session_state.email_text.strip()
    email_id = st.session_state.email_id.strip() or "DEMO-001"

    if not email_text:
        st.error("Enter email content before running the analysis.")
    else:
        action_text = (
            "If it is dangerous, quarantine it. Otherwise, explain why no quarantine is needed."
            if action_requested
            else "Analyze the email and provide a recommendation. Do not request quarantine."
        )

        prompt = f"""Analyze this email. {action_text}

Email ID: {email_id}

{email_text}
"""

        try:
            with st.spinner("The triage agent is analyzing the email..."):
                result = run_async(
                    Runner.run(
                        st.session_state.agent_system["triage_agent"],
                        prompt,
                    )
                )

            events, classifier_output = extract_trace(result)
            st.session_state.trace = events
            st.session_state.classifier_output = classifier_output
            st.session_state.final_output = result.final_output

            if getattr(result, "interruptions", None):
                request = result.interruptions[0]
                st.session_state.pending_state = result.to_state()
                st.session_state.pending_request = request
                st.session_state.pending_action = parse_jsonish(request.arguments)
                st.session_state.approval_status = "Pending human approval"
            else:
                st.session_state.approval_status = "No approval required"

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "If this is an AGNES rate-limit message, wait briefly and try again."
            )


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
has_result = (
    st.session_state.get("final_output") is not None
    or st.session_state.get("pending_request") is not None
    or st.session_state.get("classifier_output") is not None
)

if has_result:
    st.divider()
    st.subheader("Analysis Results")

    classifier = st.session_state.get("classifier_output") or {}
    classification = classifier.get("classification", "See agent result")
    probability = classifier.get("phishing_probability")
    risk = classifier.get("risk_level", "See agent result")

    if isinstance(probability, (int, float)):
        probability_display = f"{probability * 100:.2f}%"
    else:
        probability_display = "See trace"

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "ML Classification", classification),
        (c2, "Phishing Probability", probability_display),
        (c3, "Risk Level", risk),
        (c4, "Action Status", st.session_state.get("approval_status", "Complete")),
    ]
    for col, title, value in cards:
        with col:
            st.markdown(
                f'<div class="status-card"><div class="status-title">{title}</div>'
                f'<div class="status-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    # Pending HITL action
    if st.session_state.get("pending_request") is not None:
        st.warning(
            "Human approval is required before the simulated quarantine can execute.",
            icon="⚠️",
        )

        action = st.session_state.get("pending_action")
        with st.expander("Review Proposed Quarantine Action", expanded=True):
            if isinstance(action, dict):
                st.json(action)
            else:
                st.write(action)

        reject_feedback = st.text_input(
            "Optional rejection feedback",
            placeholder="Example: Needs manual verification before quarantine.",
        )

        approve_col, reject_col = st.columns(2)

        with approve_col:
            approve_clicked = st.button(
                "✅ Approve Quarantine",
                type="primary",
                use_container_width=True,
            )

        with reject_col:
            reject_clicked = st.button(
                "❌ Reject Quarantine",
                use_container_width=True,
            )

        if approve_clicked or reject_clicked:
            state = st.session_state.pending_state
            request = st.session_state.pending_request

            try:
                if approve_clicked:
                    state.approve(request)
                    decision_label = "Approved"
                else:
                    state.reject(
                        request,
                        rejection_message=(
                            "Human rejected the mock quarantine. "
                            + (reject_feedback.strip() or "No additional feedback provided.")
                        ),
                    )
                    decision_label = "Rejected"

                with st.spinner("Resuming the agent after the human decision..."):
                    resumed = run_async(
                        Runner.run(
                            st.session_state.agent_system["triage_agent"],
                            state,
                        )
                    )

                new_events, new_classifier = extract_trace(resumed)
                st.session_state.trace.extend(new_events)
                if new_classifier:
                    st.session_state.classifier_output = new_classifier

                st.session_state.final_output = resumed.final_output
                st.session_state.approval_status = decision_label
                st.session_state.pop("pending_state", None)
                st.session_state.pop("pending_request", None)
                st.session_state.pop("pending_action", None)
                st.rerun()

            except Exception as exc:
                st.error(f"Could not resume the workflow: {exc}")

    # Consolidated final response
    if st.session_state.get("final_output"):
        st.subheader("Agent Recommendation")
        st.markdown(st.session_state.final_output)

    # Trace
    if st.session_state.get("trace"):
        with st.expander("Show Agent / Tool Execution Trace"):
            for idx, event in enumerate(st.session_state.trace, start=1):
                st.markdown(
                    f"**{idx}. {event['event']} — {event['agent']}**"
                )
                if event.get("tool"):
                    st.code(event["tool"], language=None)
                details = event.get("details")
                if isinstance(details, dict):
                    st.json(details)
                elif details:
                    st.code(str(details), language=None)

st.divider()
st.caption(
    "Group 4 • ISM6422 • Classroom prototype. Synthetic data and simulated quarantine are used for demonstration."
)
