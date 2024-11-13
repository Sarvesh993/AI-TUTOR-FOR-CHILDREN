import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Path to the resources folder where your documents are stored
RESOURCE_FOLDER = 'Resource'

# Initialize ChatGroq with the API key and model
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

# Define the prompt template
prompt_teacher = PromptTemplate.from_template(
    """
    ### STUDENT QUERY:
    {student_query}
    
    ### INSTRUCTION:
    You are an AI tutor designed to help students,by understanding concepts based on the provided textbook or document. 
    Your job is to give explanations the student's questions, clarify doubts, and encourage critical thinking without giving direct answers. 
    Use the textbook content to provide step-by-step explanations, especially for subjects like Math, Physics, and Chemistry:
    
    - For **Math**, walk the student through the steps of solving problems, explaining the logic behind each step.
    - For **Physics**, discuss formulas, equations, and how they apply to real-world scenarios. Focus on explaining physical concepts.
    - For **Chemistry**, explain chemical reactions, balancing equations, and the principles behind them in detail.
    
    If the student asks about a concept, provide related examples, follow-up questions, and additional explanations. Encourage the student to think critically and engage in a discussion rather than just giving the final answer.
    When responding, avoid using phrases like "you mentioned." Directly address the student’s query and guide them through critical thinking by asking questions related to the topic.
    Your responses should feel conversational, supportive, and encourage the student to explore the topic further.
    
    ### RESPONSE:
    """
)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    retriever = vectorstore.as_retriever()
    return memory, retriever

def get_response(user_question, memory, retriever):
    # Retrieve relevant documents from the vectorstore
    retrieved_docs = retriever.get_relevant_documents(user_question)
    combined_docs = "\n".join(doc.page_content for doc in retrieved_docs)

    # Create the prompt input using the PromptTemplate, incorporating memory
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory.chat_memory.messages])
    combined_input = prompt_teacher.format(student_query=conversation_history + "\n" + user_question + "\n" + combined_docs)

    # Get response from the model
    response = llm.invoke(combined_input)

    # Store the interaction in memory
    memory.chat_memory.messages.append({"role": "user", "content": user_question})
    memory.chat_memory.messages.append({"role": "assistant", "content": response.content})

    return response.content

def handle_userinput(user_question, memory, retriever):
    response = get_response(user_question, memory, retriever)

    # Display the chat history dynamically, using memory for conversation continuity
    for message in memory.chat_memory.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(
                f"""
                <div style='background-color: #DCF8C6; color: black; padding: 10px; border-radius: 15px; margin-left: 80px; max-width: 100%;'>
                <strong>You:</strong> {content}
                </div>
                """, unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(
                f"""
                <div style='background-color: #E3F2FD; color: black; padding: 10px; border-radius: 15px; margin: 10px; max-width: 100%; margin-left: auto;' >
                <strong>AI:</strong> {content}
                </div>
                """, unsafe_allow_html=True)

def list_available_pdfs(resource_folder):
    """List all PDF files available in the resources folder."""
    return [f for f in os.listdir(resource_folder) if f.endswith(".pdf")]

def load_selected_pdfs(selected_pdfs, resource_folder):
    """Load selected PDF files from the resources folder."""
    pdf_paths = [os.path.join(resource_folder, pdf) for pdf in selected_pdfs]
    return pdf_paths

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:",layout="wide")

    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.header("Ai Tutor For Children (Tamil Nadu) :books:")
    st.caption("powered by Llama 3.1")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.memory and st.session_state.retriever:
        handle_userinput(user_question, st.session_state.memory, st.session_state.retriever)

    with st.sidebar:
        st.sidebar.info("""
            **Note:** Please select your standard and click on "Process Selected Book." 
            This may take a few minutes to complete, so kindly be patient. 
            After Processing, feel free to clear your subject-related doubts!
            """)
        st.subheader("Select a document from resources folder")

        # List available PDFs in the resources folder
        available_pdfs = list_available_pdfs(RESOURCE_FOLDER)

        # Allow users to select one or more PDFs
        selected_pdfs = st.multiselect("Choose your Book:", available_pdfs)

        if st.button("Process Selected Book"):
            if selected_pdfs:
                with st.spinner("Processing"):
                    # Load and process selected PDFs
                    pdf_paths = load_selected_pdfs(selected_pdfs, RESOURCE_FOLDER)
                    raw_text = get_pdf_text(pdf_paths)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    memory, retriever = create_conversation_chain(vectorstore)
                    st.session_state.memory = memory
                    st.session_state.retriever = retriever
            else:
                st.warning("Please select at least one document.")

if __name__ == '__main__':
    main()
