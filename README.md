# Group 4 — Agentic Phishing Email Triage Streamlit App

This is a browser-based presentation front end for the same agentic workflow used in the Group 4 project notebook.

## What the application demonstrates

- **Agent 1:** Phishing Triage Assistant
- **Agent 2:** Security Review Specialist
- **Tool 1:** `classify_email`
- **Tool 2:** `security_review`
- **Tool 3:** `quarantine_email`
- **Human-in-the-loop:** `quarantine_email` uses `needs_approval=True`
- **ML component:** TF-IDF + Logistic Regression trained on the same 126-row synthetic classroom dataset
- **Model provider:** AGNES (`agnes-3.0-flash`)
- **Framework:** OpenAI Agents SDK

The quarantine action is simulated. The application does not connect to a real email system.

---

## Recommended presentation workflow

### Before class

1. Start the Streamlit application.
2. Enter the AGNES API key in the sidebar.
3. Click **Initialize / Reset Agent System**.
4. Load **Demo 1 — Clear phishing**.
5. Keep the browser tab open.

### In class — Demo 1

1. Click **Analyze Email**.
2. The expected sequence is:
   `classify_email → security_review → quarantine_email`
3. The application should stop at **Human Approval Required**.
4. Show the proposed action.
5. Click **Approve Quarantine**.
6. The agent resumes and displays the completed mock quarantine result.

### In class — Demo 2

1. Select **Demo 2 — Legitimate but urgent-looking**.
2. Click **Load Selected Example**.
3. Click **Analyze Email**.
4. The expected result is Legitimate / Low risk with **no quarantine approval request**.

---

## Run locally or in a HiperGator terminal

From the folder containing `app.py`:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit normally prints a local URL such as `http://localhost:8501`.

> Note: If Streamlit is running on a remote HiperGator compute node, access to port 8501 may require your institution's Open OnDemand/proxy configuration. If that is inconvenient, use Streamlit Community Cloud or run the app on the presentation laptop.

---

## Streamlit Community Cloud option

The folder is deployment-ready:

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `.streamlit/config.toml`.
3. In Streamlit Community Cloud, create a new app from that repository.
4. Use `app.py` as the entry point.
5. Either:
   - enter the AGNES API key in the application's sidebar each session, or
   - add `AGNES_API_KEY` as a Streamlit secret.

Do not commit an API key into GitHub.

---

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — presentation-friendly theme
- `README.md` — these instructions

## Important

This application is an additional presentation interface. The full Jupyter notebook remains the project's main implementation/evaluation artifact.
