import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings


# Show title and description.
st.title("💬 Chat with Datadog Docs")
st.write(
    "This is a chatbot that allows you to ask questions about the [Datadog Documentation](https://docs.datadoghq.com/) \n"
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Datadog docs – hang tight! This should take 5-10 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="""You are an expert on 
            the Datadog Documentation and your 
            job is to answer technical questions. 
            Assume that all questions are related 
            to the Datadog documentation. Keep 
            your answers technical and based on 
            facts – do not hallucinate features. If something is not supported say that.""",
        )
        index = VectorStoreIndex.from_documents(docs)
        return index




# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    openai.api_key = openai_api_key
    index = load_data()

    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=True, streaming=True
        )

    if prompt := st.chat_input(
        "Ask a question about Datadog Docs!"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Write message history to UI
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)