import streamlit as st

from langchain.prompts import PromptTemplate
#from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

#import ollama
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Interview Candidate Agent V2: Ren Hwai")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_chat_input" not in st.session_state:
    st.session_state.stop_chat_input=True
if "chat_input_placeholder" not in st.session_state:
    st.session_state.chat_input_placeholder="Please update Vectorstore and LLM parameters in sidebar before begin the chat"

def update_vectorstore():
    pdf_loader=PyPDFLoader("2023 Feb Data Analyst Resume.docx.pdf")
    pdf=pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(st.session_state.chunk_size), chunk_overlap=int(st.session_state.chunk_overlap))
    splits = text_splitter.split_documents(pdf)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings())
    retriever = vectorstore.as_retriever()

    GOOGLE_API_KEY='AIzaSyBROuSzzmcJnK_IoamOPKZOhDVQjTFTCPs'
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest",google_api_key=GOOGLE_API_KEY)

    return retriever,llm

def langchain_ollama_invoke(question):
    prompt_template = PromptTemplate.from_template(st.session_state.input_prompt_template)
    description=st.session_state.description
    
    chain =(
        {
            "description":itemgetter("description"),
            "experience": itemgetter("question") | st.session_state.retriever,
            "question": itemgetter("question"),
         }
        |prompt_template
        |st.session_state.llm
        |StrOutputParser()
    )
    
    stream=chain.invoke(
        {
            "description":description.replace("\n",""),
            "experience":st.session_state.retriever,
            "question":question,
            }
            )
    for chunk in stream:
        yield chunk

#side bar
with st.sidebar:
    if st.button("Update Vectorstore and LLM Parameters"):
        st.session_state.retriever, st.session_state.llm=update_vectorstore()
        st.session_state.stop_chat_input=False
    st.session_state.description = st.text_area(
    "Paste job description in text form(optional)",
    label_visibility="visible",
    height=150,
    value="""""" 
    )
    st.session_state.input_prompt_template = st.text_area(
    "Input Prompt Template",
    label_visibility="visible",
    height=150,
    value="""You are Interview Candidate Agent.  Your name is Kong Ren Hwai. You are interviewing for job position with job description found below. You will answer question based on context and fit into the description. You will not used the words in blacklist. If the question is asking about private personal information, reply "Please refer to the resume for private information, thank you."
    job description: {description}
    experience: {experience}
    question: {question}
    blacklist: {{As Kong Ren Hwai}}
    """ 
    )
    st.session_state.chunk_size = st.number_input(
    "Chunk Size",
    label_visibility="visible",
    value=1000
    )
    st.session_state.chunk_overlap = st.number_input(
    "Chunk Overlap",
    label_visibility="visible",
    value=200
    )
    st.session_state.max_token = st.number_input(
    "Max token generate. Larger token number, longer text generated. (Max 4096)",
    label_visibility="visible",
    value=4096
    )
    st.session_state.temperature = st.number_input(
    "Temperature (min:0.0, max:1.0). Higher temperature, more creative response",
    label_visibility="visible",
    value=0.3
    )
    st.session_state.top_k = st.number_input(
    "Top K. Higher number, more diverse response",
    label_visibility="visible",
    value=0
    )

#Main chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if st.session_state.stop_chat_input is False:
        st.session_state.chat_input_placeholder="Hiring manager, please start the job interview..."

if prompt := st.chat_input(st.session_state.chat_input_placeholder,disabled=st.session_state.stop_chat_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response=st.write_stream(langchain_ollama_invoke(prompt))
    st.session_state.messages.append(
        {"role": "assistant", "content": response})

