import json
import os
import uuid
 
import boto3
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
ARN = os.environ["AGENT_RUNTIME_ARN"].strip()
REGION = os.getenv("REGION_NAME", "us-east-1").strip()
 
 
@st.cache_resource
def agentcore_client():
    return boto3.client("bedrock-agentcore", region_name=REGION)
 
 
def ask_agent(prompt: str, user_id: str, session_id: str) -> str:
    resp = agentcore_client().invoke_agent_runtime(
        agentRuntimeArn=ARN,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": prompt, "user_id": user_id}),
    )
    body = json.loads(resp["response"].read())
    answer = body.get("result") or body.get("error") or str(body)
    if body.get("tools_used"):
        answer += f"\n\n_tools used: {', '.join(body['tools_used'])}_"
    for err in body.get("tool_errors") or []:
        answer += f"\n\n:red[**tool error** {err}]"
    return answer
 
 
st.set_page_config(page_title="Education Assistant", page_icon="🎓")
st.title("🎓 Education Assistant")
 
with st.sidebar:
    user_id = st.text_input("Your name / student id", value="dat").strip() or "anonymous"
    if st.button("New conversation"):
        st.session_state.pop("session_id", None)
        st.session_state.pop("messages", None)
    st.caption("Tutor · Study planner · Research papers")
 
if "session_id" not in st.session_state:
    st.session_state.session_id = f"{user_id}-session-{uuid.uuid4().hex}" 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
if prompt := st.chat_input("Ask about math, English, your schedule, papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_agent(prompt, user_id, st.session_state.session_id)
            except Exception as e:
                answer = f"⚠️ Error calling the agent: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})