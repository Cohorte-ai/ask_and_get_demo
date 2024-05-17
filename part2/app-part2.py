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

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)

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
        query_options = ["None", "Do we have a paper discussing evaluation of LLM Generations with a panel of diverse models",
                         "I need information on hallucinations in multi-model llms"]
        # emojis = ["👥", "📅", "🏷️"]
        selected_query = pills(
            "Select example queries or enter your own query in the chat input below", query_options, key="query_pills", clearable=True)

        def add_to_message_history(role, content):
            # print(content)
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
                # st.write(docs)
                docs[0].doc_id = text_file.stem  # Using the stem of the file name as the document ID
                arxiv_docs.extend(docs)
            
            return arxiv_docs

        # To be executed only once
        # wiki_titles = ["Mumbai", "Chennai", "Toronto", "Seattle", "Chicago", "Boston", "Houston"]
        # city_documents = fetch_and_load_wiki_documents(wiki_titles)

        # # To be executed only once
        # arxiv_documents = process_pdf_documents()
        # for docs in arxiv_documents:
        #     # docs.metadata['creation_date'] = '01/01/1989'
        #     file_name = docs.metadata['file_name']
        #     year = '20' + file_name[:2]
        #     month = file_name[2:4]
        #     docs.metadata['creation_date'] = f"{month}/{year}"


        def build_document_summary_index(docs):
            from llama_index.core.node_parser import TokenTextSplitter

            node_parser = TokenTextSplitter(
                separator=" ", chunk_size=2048, chunk_overlap=200
            )
            orig_nodes = node_parser.get_nodes_from_documents(arxiv_documents)
            # print(len(orig_nodes))

            from llama_index.core.extractors import (
                SummaryExtractor,
                QuestionsAnsweredExtractor,
                KeywordExtractor
            )
            from llama_index.core.schema import MetadataMode


            extractors_1 = [
                QuestionsAnsweredExtractor(
                    questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
                ),
                KeywordExtractor(keywords=10, llm=llm),
            ]


            extractors_2 = [
                SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
                QuestionsAnsweredExtractor(
                    questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
                ),
                KeywordExtractor(keywords=10, llm=llm),

            ]

            nodes = orig_nodes[1:4]

            from llama_index.core.ingestion import IngestionPipeline

            # process nodes with metadata extractors
            pipeline = IngestionPipeline(transformations=[node_parser, *extractors_2])

            nodes_2 = pipeline.run(nodes=orig_nodes, in_place=False, show_progress=True)

            # print(nodes_2[2].get_content(metadata_mode="all"))
            from llama_index.core import VectorStoreIndex

            index = VectorStoreIndex(nodes_2)
            
            return index

        # #To be executed only once.
        # document_index = build_document_summary_index(arxiv_documents)
        # document_index.storage_context.persist("meta")

        storage_context = StorageContext.from_defaults(persist_dir='meta')


        def load_index(index_name):           
            # Rebuild storage context from defaults with the persist directory
            storage_context = StorageContext.from_defaults(persist_dir=index_name)
            
            # Reload the document summary index from the storage
            doc_summary_index = load_index_from_storage(storage_context)
            
            return doc_summary_index

        document_index = load_index("meta")

        from llama_index.core.retrievers import QueryFusionRetriever

        retriever = document_index.as_retriever(similarity_top_k=10)

        # nodes_with_scores = retriever.retrieve(
        #     "tell me about phi3 and mixture of depths?"
        # )

        # for node in nodes_with_scores:
        #     st.code(node.metadata)


        # retriever = DocumentSummaryIndexLLMRetriever(
        #     doc_summary_index,
        #     # choice_select_prompt=None,
        #     choice_batch_size=10,
        #     choice_top_k=3,
        #     # format_node_batch_fn=None,
        #     # parse_choice_select_answer_fn=None,
        # )


        
        # from llama_index.core.retrievers import AutoMergingRetriever
        # base_retriever = doc_summary_index.as_retriever(similarity_top_k=20)
        # retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
        
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
            # response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
            st.session_state["query_engine"] = document_index.as_query_engine(similarity_top_k=3)

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
                    import re

                    # Define a regex pattern to match dates in the format MM/YYYY
                    date_pattern = r'\b\d{2}/\d{4}\b'

                    # Search for the pattern in the prompt
                    match = re.search(date_pattern, selected_query)

                    # If a match is found, extract the date
                    if match:
                        extracted_date = match.group(0)
                        print(f"Extracted date: {extracted_date}")

                        # Filter the retrieved_nodes to keep only those with a matching creation_date
                        filtered_nodes = [node for node in retrieved_nodes if node.metadata['creation_date'] == extracted_date]
                    else:
                        print("No date found in the prompt.")
                        filtered_nodes = retrieved_nodes

                    # st.write(filtered_nodes)
                    # Initialize a dictionary to hold the document names and concatenated summaries
                    doc_summary_by_name = {}
                    doc_keywords_by_name = {}

                    # Loop through each node in the retrieved_nodes
                    for node in filtered_nodes:
                        # st.code(node.get_content(metadata_mode="all"))
                        # st.code(node.metadata['excerpt_keywords'])
                        # Get the document name from node metadata
                        doc_name = node.metadata['file_name']
                        doc_keywords = node.metadata['excerpt_keywords']
                        
                        # Get the document summary
                        doc_summary = node.metadata['section_summary']
                        
                        # Check if the document name already exists in the dictionary
                        if doc_name in doc_summary_by_name:
                            # Concatenate the new summary to the existing summary for this document name
                            print("Summary already present")
                            doc_summary_by_name[doc_name] += doc_summary
                            doc_keywords_by_name[doc_name] += doc_keywords

                        else:
                            # Otherwise, add the document name and summary to the dictionary
                            doc_summary_by_name[doc_name] = doc_summary
                            doc_keywords_by_name[doc_name] = doc_keywords
                    # st.write(doc_summary_by_name)

                    # Optional: If you want to output this information
                    # Initialize an empty list to store each final response
                    responses = []

                    for doc_name, summaries in doc_summary_by_name.items():
                        text = f"""
**Document Name**: {os.path.splitext(doc_name)[0]}  
**Document Summary**: {summaries} \n
**Document Keywords**: {doc_keywords_by_name[doc_name]}"""
                        # st.code(text)
                        llmresponse = llm.complete(f"""
                            User asked a query, for which our AI application has suggested a document which contains relevant information for the user query. The document's name, summary and keywords are provided below in card markdown:
                            
                            User query: {selected_query}

                            card markdown: {text}

                            Based on the above information, first check if the document is relevant to the user query. Only if it is absolutely relevant, then provide a justification as to why the above document is relevant to the user query. Otherwise reply saying that document is irrelevant

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

                    # st.markdown(combinedfinalresponse)
                    # st.markdown(response.source_nodes)

                    # Add to message history or perform other actions with the combined response
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


                    # response = st.session_state["query_engine"].query(prompt)
                    # print(response)
                    # print(response.metadata)

                    
                    retrieved_nodes = retriever.retrieve(prompt)
                    
                    import re

                    # Define a regex pattern to match dates in the format MM/YYYY
                    date_pattern = r'\b\d{2}/\d{4}\b'

                    # Search for the pattern in the prompt
                    match = re.search(date_pattern, prompt)

                    # If a match is found, extract the date
                    if match:
                        extracted_date = match.group(0)
                        print(f"Extracted date: {extracted_date}")

                        # Filter the retrieved_nodes to keep only those with a matching creation_date
                        filtered_nodes = [node for node in retrieved_nodes if node.metadata['creation_date'] == extracted_date]
                    else:
                        print("No date found in the prompt.")
                        filtered_nodes = retrieved_nodes

                    # st.write(filtered_nodes)
                    # Initialize a dictionary to hold the document names and concatenated summaries
                    doc_summary_by_name = {}
                    doc_keywords_by_name = {}

                    # Loop through each node in the retrieved_nodes
                    for node in filtered_nodes:
                        # st.code(node.get_content(metadata_mode="all"))
                        # st.code(node.metadata['excerpt_keywords'])
                        # Get the document name from node metadata
                        doc_name = node.metadata['file_name']
                        doc_keywords = node.metadata['excerpt_keywords']
                        
                        # Get the document summary
                        doc_summary = node.metadata['section_summary']
                        
                        # Check if the document name already exists in the dictionary
                        if doc_name in doc_summary_by_name:
                            # Concatenate the new summary to the existing summary for this document name
                            print("Summary already present")
                            doc_summary_by_name[doc_name] += doc_summary
                            doc_keywords_by_name[doc_name] += doc_keywords

                        else:
                            # Otherwise, add the document name and summary to the dictionary
                            doc_summary_by_name[doc_name] = doc_summary
                            doc_keywords_by_name[doc_name] = doc_keywords
                    # st.write(doc_summary_by_name)

                    # Optional: If you want to output this information
                    # Initialize an empty list to store each final response
                    responses = []

                    for doc_name, summaries in doc_summary_by_name.items():
                        text = f"""
**Document Name**: {os.path.splitext(doc_name)[0]}  
**Document Summary**: {summaries} \n
**Document Keywords**: {doc_keywords_by_name[doc_name]}"""
                        # st.code(text)
                        llmresponse = llm.complete(f"""
                            User asked a query, for which our AI application has suggested a document which contains relevant information for the user query. The document's name, summary and keywords are provided below in card markdown:
                            
                            User query: {prompt}

                            card markdown: {text}

                            Based on the above information, first check if the document is relevant to the user query. Only if it is absolutely relevant, then provide a justification as to why the above document is relevant to the user query. Otherwise reply saying that document is irrelevant

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

                    # st.markdown(combinedfinalresponse)
                    # st.markdown(response.source_nodes)

                    # Add to message history or perform other actions with the combined response
                    add_to_message_history("assistant", combinedfinalresponse)



if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()