import streamlit as st
import os
import openai
import nest_asyncio
nest_asyncio.apply()

import PyPDF2

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
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

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
        query_options = ["None", "I need information about Transformer models and how infinite context can be processed.",
                         "I need details about scaling rectified flow for image synthesis"]
        # emojis = ["👥", "📅", "🏷️"]
        selected_query = pills(
            "Select example queries or enter your own query in the chat input below", query_options, key="query_pills", clearable=True)

        def add_to_message_history(role, content):
            print(content)
            message = {"role": role, "content": "\n\n"+str(content)}
            st.session_state["messages"].append(message)  # Add response to message history
        

        # class CustomDirectoryReader(SimpleDirectoryReader):
        #     def __init__(self, return_full_document=False, **kwargs):
        #         super().__init__(**kwargs)
        #         self.return_full_document = return_full_document
        #         # Assuming PDFReader supports a return_full_document parameter
        #         self.file_extractor[".pdf"] = PDFReader(return_full_document=self.return_full_document)

        # def load_documents(path_to_directory):
        #     reader = CustomDirectoryReader(input_dir=path_to_directory, return_full_document=True)
        #     documents = reader.load_data()
        #     return documents

        def process_pdf_documents():
            # Ensure the output directory exists
            data_path = Path("txt")
            data_path.mkdir(exist_ok=True)
            
            # Directory containing the PDFs
            pdf_directory = Path("pdfs")
            
            # Process each PDF in the directory
            for pdf_file in pdf_directory.glob("*.pdf"):
                # Open the PDF file
                with open(pdf_file, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = []
                    
                    # Read each page in the PDF
                    for page in reader.pages:
                        text.append(page.extract_text())
                    
                    # Join all text from all pages
                    full_text = "\n".join(filter(None, text))
                
                # Save the extracted text to a .txt file with the same name
                output_file_path = data_path / f"{pdf_file.stem}.txt"
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(full_text)
            
            # Load all text documents as arxiv_docs
            arxiv_docs = []
            for text_file in data_path.glob("*.txt"):
                docs = SimpleDirectoryReader(
                    input_files=[str(text_file)]
                ).load_data()
                docs[0].doc_id = text_file.stem  # Using the stem of the file name as the document ID
                arxiv_docs.extend(docs)
            
            return arxiv_docs

        # To be executed only once
        # wiki_titles = ["Mumbai", "Chennai", "Toronto", "Seattle", "Chicago", "Boston", "Houston"]
        # city_documents = fetch_and_load_wiki_documents(wiki_titles)

        # To be executed only once
        arxiv_documents = process_pdf_documents()
        # st.write(arxiv_documents)

        from llama_index.core.node_parser import (
            HierarchicalNodeParser,
            SentenceSplitter,
        )
        node_parser = HierarchicalNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(arxiv_documents)
        print("How many nodes")
        print(len(nodes))

        from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
        leaf_nodes = get_leaf_nodes(nodes)
        print("leaf nodes")
        print(len(leaf_nodes))

        # define storage context
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core import StorageContext

        docstore = SimpleDocumentStore()

        # insert nodes into docstore
        docstore.add_documents(nodes)

        # define storage context (will include vector store by default too)
        storage_context = StorageContext.from_defaults(docstore=docstore)

        ## Load index into vector index
        from llama_index.core import VectorStoreIndex

        base_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        from llama_index.core.retrievers import AutoMergingRetriever
        base_retriever = base_index.as_retriever(similarity_top_k=6)
        retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

        query_str = (
            "I need information about transformers"
            "I also need information about stable diffusion 3"
        )

        nodes = retriever.retrieve(query_str)
        base_nodes = base_retriever.retrieve(query_str)
        st.write(len(nodes))
        st.write(len(base_nodes))

        from llama_index.core.response.notebook_utils import display_source_node

        for node in nodes:
            st.write(node)
            st.write("METAAAADATA")
            st.write(node.metadata)

        from llama_index.core.response.notebook_utils import display_source_node

        for node in base_nodes:
            # display_source_node(node, source_length=10000)
            st.write(node)
            st.write("METAAAADATA")
            st.write(node.metadata)


        def build_document_summary_index(docs):
            # Initialize the LLM (gpt-3.5-turbo) with OpenAI
            
            # Initialize the SentenceSplitter for splitting long sentences
            splitter = SentenceSplitter(chunk_size=1024)
            
            # Setup the response synthesizer with the specified mode
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize", use_async=True
            )
            
            # Build the document summary index
            doc_summary_index = DocumentSummaryIndex.from_documents(
                docs,
                llm=llm,
                transformations=[splitter],
                response_synthesizer=response_synthesizer,
                show_progress=True,
            )
            
            return doc_summary_index

        #To be executed only once.
        # document_index = build_document_summary_index(arxiv_documents)
        # document_index.storage_context.persist("arxivindexfull")


        def load_index(index_name):           
            # Rebuild storage context from defaults with the persist directory
            storage_context = StorageContext.from_defaults(persist_dir=index_name)
            
            # Reload the document summary index from the storage
            doc_summary_index = load_index_from_storage(storage_context)
            
            return doc_summary_index

        doc_summary_index = load_index("arxivindexfull")

        retriever = DocumentSummaryIndexLLMRetriever(
            doc_summary_index,
            # choice_select_prompt=None,
            choice_batch_size=10,
            choice_top_k=3,
            # format_node_batch_fn=None,
            # parse_choice_select_answer_fn=None,
        )

        
        # retriever = DocumentSummaryIndexEmbeddingRetriever(
        #     doc_summary_index,
        #     similarity_top_k=3,
        # )

       # Sidebar header
        st.sidebar.markdown("## File Repository Viewer")
        pdf_folder_path = "pdfs"

        # Check if the directory exists
        if os.path.exists(pdf_folder_path):
            # List all files in the 'pdfs' folder
            files = os.listdir(pdf_folder_path)
            
            # Create a DataFrame from the list of files
            df_files = pd.DataFrame(files, columns=['File Names'])

            # Set DataFrame index starting from 1 instead of 0
            df_files.index = range(1, len(df_files) + 1)
            
            # Display the DataFrame in the sidebar
            st.sidebar.dataframe(df_files.style.set_properties(**{'text-align': 'left', 'white-space': 'normal'}))
        else:
            st.sidebar.write("The 'pdfs' folder does not exist.")


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
                st.markdown(f"""
                <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; border: 1px solid #e1e1e1;">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

        # To avoid duplicated display of answered pill questions each rerun
        if selected_query and selected_query != "None" and selected_query not in st.session_state.get(
            "displayed_pill_questions", set()
        ):
            with st.spinner():
                st.session_state.setdefault("displayed_pill_questions", set()).add(selected_query)
                with st.chat_message("user"):
                    st.write(selected_query)
                with st.chat_message("assistant"):
                    retrieved_nodes = retriever.retrieve(selected_query)
                    doc_summary_by_name = {}
                    # Loop through each node in the retrieved_nodes
                    for node in retrieved_nodes:
                        # Get the document name from node metadata
                        doc_name = node.metadata['file_name']
                        
                        # Get the document summary
                        doc_summary = f""" {doc_summary_index.get_document_summary(node.node.ref_doc_id)}
                        
                        """
                        # Check if the document name already exists in the dictionary
                        if doc_name in doc_summary_by_name:
                            # Concatenate the new summary to the existing summary for this document name
                            print("Summary already present")
                        else:
                            # Otherwise, add the document name and summary to the dictionary
                            doc_summary_by_name[doc_name] = doc_summary                    
                    # Initialize an empty list to store each final response
                    responses = []

                    for doc_name, summaries in doc_summary_by_name.items():
                        text = f"""
**Document Name**: {os.path.splitext(doc_name)[0]}  
**Document Summary**: {summaries}"""

                        llmresponse = llm.complete(f"""
                            User asked a query, for which our AI application has suggested a document which contains relevant information for the user query. The document's name and summary is provided below in card markdown:
                            
                            card markdown: {text}

                            Based on the above information, provide a justification as to why the above document is relevant to the user query. Return just the justification text and nothing else.

                            """)
                        # Assume llmresponse.text gives the justification text directly

                        finalresponse = f"{text.rstrip()}  \n\n**Justification**: {llmresponse.text}"
                        responses.append(finalresponse)
                        st.markdown(f"""
                        <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; border: 1px solid #e1e1e1;">
                            {finalresponse}
                        </div>
                        """, unsafe_allow_html=True)

                    # Combine all responses into one formatted string
                    combinedfinalresponse = "\n\n---\n\n".join(responses)
                    add_to_message_history("user", selected_query)
                    add_to_message_history("assistant", combinedfinalresponse)

        # Prompt for user input and save to chat history
        prompt = st.chat_input("Enter your natural language query about the Arxiv paper collection") 
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
                    # response_text = "| Document Name | Document Summary |\n|------------|---------------|\n"

                    # Loop over each node in retrieved_nodes to append each node's score and metadata to the table
                    # Assume we have the retrieved_nodes list populated as required

                    # Initialize a dictionary to hold the document names and concatenated summaries
                    doc_summary_by_name = {}

                    # Loop through each node in the retrieved_nodes
                    for node in retrieved_nodes:
                        # st.write(node)
                        # Get the document name from node metadata
                        doc_name = node.metadata['file_name']
                        
                        # Get the document summary
                        doc_summary = f""" {doc_summary_index.get_document_summary(node.node.ref_doc_id)}
                        
                        """
                        
                        # Check if the document name already exists in the dictionary
                        if doc_name in doc_summary_by_name:
                            # Concatenate the new summary to the existing summary for this document name
                            print("Summary already present")
                        else:
                            # Otherwise, add the document name and summary to the dictionary
                            doc_summary_by_name[doc_name] = doc_summary
                    

                    # Optional: If you want to output this information
                    # Initialize an empty list to store each final response
                    responses = []

                    for doc_name, summaries in doc_summary_by_name.items():
                        text = f"""
**Document Name**: {os.path.splitext(doc_name)[0]}  
**Document Summary**: {summaries}"""

                        llmresponse = llm.complete(f"""
                            User asked a query, for which our AI application has suggested a document which contains relevant information for the user query. The document's name and summary is provided below in card markdown:
                            
                            card markdown: {text}

                            Based on the above information, provide a justification as to why the above document is relevant to the user query. Return just the justification text and nothing else.

                            """)
                        # Assume llmresponse.text gives the justification text directly

                        finalresponse = f"{text.rstrip()}  \n\n**Justification**: {llmresponse.text}"
                        responses.append(finalresponse)
                        st.markdown(f"""
                        <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; border: 1px solid #e1e1e1;">
                            {finalresponse}
                        </div>
                        """, unsafe_allow_html=True)
                        # st.info("---")

                    # Combine all responses into one formatted string
                    combinedfinalresponse = "\n\n---\n\n".join(responses)

                    # Add to message history or perform other actions with the combined response
                    add_to_message_history("assistant", combinedfinalresponse)



if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()