import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('Common questions asked during pregnancy')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)

load_dotenv()

def main():
    st.header("Pregnancy FAQ Chatbot💬")

    # Get the file path from the user
    file_path = "Common-Questions-in-Pregnancy-pdf.pdf"

    # Check if a file path is provided
    if file_path:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Extracting store name from the file path
            store_name = os.path.splitext(os.path.basename(file_path))[0]
            st.write(f'{store_name}')

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            # Check if "messages" not in st.session_state
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

            # Display chat messages
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")

            # Check if the user has asked a question
            if query:
                # Append user message to session_state.messages
                st.session_state.messages.append({"role": "user", "content": query})
                st.chat_message("user").write(query)

                # Perform chatbot response
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)

                msg = response
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)

if __name__ == '__main__':
    main()



# import openai
# import streamlit as st

# # Sidebar contents
# with st.sidebar:
#     st.title('Common questions asked during pregnancy')
#     st.markdown('''
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](https://streamlit.io/)
#     - [LangChain](https://python.langchain.com/)
#     - [OpenAI](https://platform.openai.com/docs/models) LLM model
#     ''')
#     add_vertical_space(5)
# def main():
#     st.title("💬 Chatbot")
#     st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])

#     if prompt := st.chat_input():
    
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user").write(prompt)
#         response = 'Results'
#         msg = response
#         st.session_state.messages.append(msg)
#         st.chat_message("assistant").write(msg)

# if __name__ == '__main__':
#     main()
