import streamlit as st
import requests


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Dynamic AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# FASTAPI CONFIGURATION
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000/chat"


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = 9

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = 8


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🤖 Dynamic AI Chatbot")

st.write(
    "AI-powered customer support chatbot using "
    "NLP, FAQ matching, sentiment analysis, "
    "conversation memory, Gemini and MySQL."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Chat Settings")

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=st.session_state.user_id,
        step=1
    )

    conversation_id = st.number_input(
        "Conversation ID",
        min_value=1,
        value=st.session_state.conversation_id,
        step=1
    )

    st.session_state.user_id = user_id
    st.session_state.conversation_id = conversation_id

    st.divider()

    st.subheader("System")

    st.success("FastAPI Backend: Connected")

    st.write(
        f"API: `{API_URL}`"
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            if "intent" in message:

                st.caption(
                    f"Intent: {message['intent']} | "
                    f"Sentiment: {message['sentiment']} | "
                    f"Source: {message['response_source']}"
                )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

user_message = st.chat_input(
    "Type your message here..."
)


# ---------------------------------------------------------
# PROCESS USER MESSAGE
# ---------------------------------------------------------

if user_message:

    # Display user message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):
        st.write(user_message)

    # Call FastAPI
    try:

        with st.spinner("Thinking..."):

            response = requests.post(
                API_URL,
                json={
                    "user_id": int(user_id),
                    "conversation_id": int(conversation_id),
                    "message": user_message
                },
                timeout=60
            )

        # Check HTTP status
        if response.status_code == 200:

            result = response.json()

            bot_response = result.get(
                "response",
                "Sorry, I could not generate a response."
            )

            intent = result.get(
                "intent",
                "unknown"
            )

            sentiment = result.get(
                "sentiment",
                "Neutral"
            )

            response_source = result.get(
                "response_source",
                "Unknown"
            )

            response_time = result.get(
                "response_time",
                0
            )

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": bot_response,
                    "intent": intent,
                    "sentiment": sentiment,
                    "response_source": response_source
                }
            )

            # Display assistant response
            with st.chat_message("assistant"):

                st.write(bot_response)

                st.caption(
                    f"Intent: {intent} | "
                    f"Sentiment: {sentiment} | "
                    f"Source: {response_source} | "
                    f"Response time: {response_time:.2f}s"
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI server."
        )

        st.info(
            "Please make sure FastAPI is running with:\n\n"
            "`uvicorn app.api:app --reload`"
        )

    except requests.exceptions.Timeout:

        st.error(
            "The chatbot request timed out."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )