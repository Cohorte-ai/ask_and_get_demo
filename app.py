import streamlit as st
import os
import openai
import nest_asyncio
nest_asyncio.apply()

from typing import Dict, Any, List

from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

#dependencies for downloading files
from pathlib import Path
import requests

#Dependencies for storage
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

from llama_index.core import Settings

from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)

from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.llama_pack import BaseLlamaPack
import pandas as pd
from streamlit_pills import pills
from PIL import Image

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


class StreamlitChatPack(BaseLlamaPack):

    def __init__(
        self,
        page: str = "AI Document Navigator",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""

        self.page = page

    def load_css(self, file_name):
        with open(file_name) as css:
            st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import streamlit as st

        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        # Load your CSS
        self.load_css('style.css')

        # Load your image
        image = Image.open('logo.png')
        # Display the image in the sidebar at the top left
        st.sidebar.image(image, width=40)

        if "messages" not in st.session_state:  # Initialize the chat messages history
            st.session_state["messages"] = [
                {"role": "assistant",
                    "content": f"Hello. Which documents do you want to search for?"}
            ]

        st.title(
            f"{self.page}💬"
        )
        st.info(
            f"Explore your document repository with this AI-powered app. Pose any natural language query about the information you want and receive a list of matched documents, their summary and the rationale behind the match.",
            icon="ℹ️",
        )
        # Define the pills with emojis
        query_options = ["None", "Option 1",
                         "Option 2"]
        # emojis = ["👥", "📅", "🏷️"]
        selected_query = pills(
            "Select example queries or enter your own query in the chat input below", query_options, key="query_pills", clearable=True)

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(message)  # Add response to message history

        def fetch_and_load_wiki_documents(city_names: List[str]):
            # Ensure the data directory exists
            data_path = Path("data")
            data_path.mkdir(exist_ok=True)

            # Fetch and save Wikipedia extracts for each city
            for city_name in city_names:
                response = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "titles": city_name,
                        "prop": "extracts",
                        "explaintext": True,
                    },
                ).json()
                page = next(iter(response["query"]["pages"].values()))
                wiki_text = page["extract"]

                with open(data_path / f"{city_name}.txt", "w") as fp:
                    fp.write(wiki_text)

            # Load all wiki documents
            city_docs = []
            for city_name in city_names:
                docs = SimpleDirectoryReader(
                    input_files=[f"data/{city_name}.txt"]
                ).load_data()
                docs[0].doc_id = city_name
                city_docs.extend(docs)
            
            return city_docs

        # To be executed only once
        # wiki_titles = ["Mumbai", "Chennai", "Toronto", "Seattle", "Chicago", "Boston", "Houston"]
        # city_documents = fetch_and_load_wiki_documents(wiki_titles)

        def build_document_summary_index(city_docs):
            # Initialize the LLM (gpt-3.5-turbo) with OpenAI
            llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
            
            # Initialize the SentenceSplitter for splitting long sentences
            splitter = SentenceSplitter(chunk_size=1024)
            
            # Setup the response synthesizer with the specified mode
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize", use_async=True
            )
            
            # Build the document summary index
            doc_summary_index = DocumentSummaryIndex.from_documents(
                city_docs,
                llm=llm,
                transformations=[splitter],
                response_synthesizer=response_synthesizer,
                show_progress=True,
            )
            
            return doc_summary_index

        #To be executed only once.
        # document_index = build_document_summary_index(city_documents)
        # document_index.storage_context.persist("index")


        def load_index(index_name):           
            # Rebuild storage context from defaults with the persist directory
            storage_context = StorageContext.from_defaults(persist_dir=index_name)
            
            # Reload the document summary index from the storage
            doc_summary_index = load_index_from_storage(storage_context)
            
            return doc_summary_index

        doc_summary_index = load_index("index")

        # retriever = DocumentSummaryIndexLLMRetriever(
        #     doc_summary_index,
        #     # choice_select_prompt=None,
        #     # choice_batch_size=10,
        #     # choice_top_k=1,
        #     # format_node_batch_fn=None,
        #     # parse_choice_select_answer_fn=None,
        # )

        
        retriever = DocumentSummaryIndexEmbeddingRetriever(
            doc_summary_index,
            # similarity_top_k=1,
        )

       # Sidebar for database schema viewer
        st.sidebar.markdown("## File Repository Viewer")

        """
        Load file repo here
        """

        st.sidebar.markdown('## Disclaimer')
        st.sidebar.markdown(
            """This application is for demonstration purposes only and may not cover all aspects of real-world data complexities. Please use it as a guide and not as a definitive source for decision-making.""")
        
        if "query_engine" not in st.session_state:  # Initialize the query engine
            response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
            st.session_state["query_engine"] = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # To avoid duplicated display of answered pill questions each rerun
        if selected_query and selected_query != "None" and selected_query not in st.session_state.get(
            "displayed_pill_questions", set()
        ):
            with st.spinner():
                st.session_state.setdefault("displayed_pill_questions", set()).add(selected_query)
                with st.chat_message("user"):
                    st.write(selected_query)
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query(
                        "User Question:"+selected_query+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    add_to_message_history("user", selected_query)
                    add_to_message_history("assistant", sql_query)

        # Prompt for user input and save to chat history
        prompt = st.chat_input("Enter your natural language query about the database") 
        if prompt:
            with st.spinner():
                add_to_message_history("user", prompt)

                # Display the new question immediately after it is entered
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    # response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                    retrieved_nodes = retriever.retrieve(prompt)

                    # sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    # Initialize a string with the Markdown table headers
                    response_text = "| Node Score | Node Metadata |\n|------------|---------------|\n"

                    # Loop over each node in retrieved_nodes to append each node's score and metadata to the table
                    for node in retrieved_nodes:
                        print(node)
                        # Append each node's score and metadata to the Markdown table
                        response_text += f"| {node.metadata['file_name']} | {node.metadata} |\n"

                    # The response_text now contains the Markdown table with all nodes' scores and metadata
                    print(response_text)
                    
                    response_container = st.empty()
                    response_container.write(response_text)
                    add_to_message_history("assistant", response_text)


if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()