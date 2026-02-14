"""
Streamlit Chat Interface for Epstein Docs Agent API
"""

import os
import streamlit as st
import requests
import json
import uuid

# Default values from environment
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:3002")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")

# Page config
st.set_page_config(
    page_title="Epstein Docs Agent",
    page_icon="🔍",
    layout="centered"
)

st.title("Epstein Docs Agent Chat")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="URL de l'API (local ou production)"
    )
    st.session_state.api_url = api_url

    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", DEFAULT_API_KEY),
        type="password",
        help="Clé API pour l'authentification"
    )
    st.session_state.api_key = api_key

    st.divider()

    # Session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.text(f"Session: {st.session_state.session_id[:8]}...")

    if st.button("Nouvelle conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Test connection
    if st.button("Tester la connexion"):
        try:
            response = requests.get(f"{api_url}/", timeout=5)
            if response.status_code == 200:
                st.success("Connexion OK")
                stats_response = requests.get(f"{api_url}/api/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    st.info(f"Documents: {stats.get('total_documents', 'N/A')}")
            else:
                st.error(f"Erreur: {response.status_code}")
        except Exception as e:
            st.error(f"Connexion échouée: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show tools used
        if message["role"] == "assistant" and "tools" in message and message["tools"]:
            with st.expander(f"🔧 Outils utilisés ({len(message['tools'])})"):
                for tool in message["tools"]:
                    st.markdown(f"**{tool['name']}**")
                    st.code(json.dumps(tool["args"], indent=2, ensure_ascii=False), language="json")
        # Show sources
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"📄 Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.code(source)

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request
    headers = {"Content-Type": "application/json"}
    if st.session_state.api_key:
        headers["X-API-Key"] = st.session_state.api_key

    payload = {
        "question": prompt,
        "session_id": st.session_state.session_id
    }

    # Get streaming response
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{st.session_state.api_url}/api/query/stream",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
            else:
                # Containers for dynamic content
                tool_container = st.container()
                message_placeholder = st.empty()

                full_response = ""
                all_sources = []
                all_tools = []
                current_tool = None

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            try:
                                event = json.loads(line_str[6:])
                                event_type = event.get("type", "")

                                if event_type == "token":
                                    content = event.get("content", "")
                                    full_response += content
                                    message_placeholder.markdown(full_response + "▌")

                                elif event_type == "tool_start":
                                    tool_name = event.get("tool_name", "unknown")
                                    tool_args = event.get("tool_args", {})
                                    current_tool = {"name": tool_name, "args": tool_args}
                                    all_tools.append(current_tool)

                                    # Display tool being called
                                    with tool_container:
                                        st.info(f"🔧 Appel: **{tool_name}**")

                                elif event_type == "tool_end":
                                    tool_name = event.get("tool_name", "unknown")
                                    sources = event.get("sources", [])

                                    # Collect sources
                                    for src in sources:
                                        if src not in all_sources:
                                            all_sources.append(src)

                                    with tool_container:
                                        if sources:
                                            st.success(f"✓ {tool_name} - {len(sources)} doc(s) trouvé(s)")
                                        else:
                                            st.success(f"✓ {tool_name}")

                                elif event_type == "error":
                                    st.error(f"Erreur: {event.get('content', 'Unknown error')}")
                                    break

                            except json.JSONDecodeError:
                                continue

                # Final display
                message_placeholder.markdown(full_response)

                # Show tools summary
                if all_tools:
                    with st.expander(f"🔧 Outils utilisés ({len(all_tools)})"):
                        for tool in all_tools:
                            st.markdown(f"**{tool['name']}**")
                            st.code(json.dumps(tool["args"], indent=2, ensure_ascii=False), language="json")

                # Show sources summary
                if all_sources:
                    with st.expander(f"📄 Sources ({len(all_sources)})"):
                        for source in all_sources:
                            st.code(source)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": all_sources,
                    "tools": all_tools
                })

        except requests.exceptions.Timeout:
            st.error("Timeout - la requête a pris trop de temps")
        except Exception as e:
            st.error(f"Erreur: {e}")
