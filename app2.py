from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader #allows to take the text out of the pdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain_community.vectorstores import FAISS
import os

os.environ['OPENAI_API_KEY'] = #your api key

# Your code here

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF ")
    
    #upload the file
    pdf=st.file_uploader("Upload your PDF", type="pdf")

    #extract the text from the file
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        #now loop through the pages first it will take the text and then the 
        #text out of the pages initialize an empty string
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)

        #create embeddings
        embeddings=OpenAIEmbeddings() #with this embedding we can create an object which we can search
        knowledge_base = FAISS.from_texts(chunks, embeddings)#used for serching in the document performs semantic/similarity search
        #show user input
        user_question=st.text_input("Ask a question about the PDF:")
        if user_question:#docs contain /are chunks which have info about user's query
            docs=knowledge_base.similarity_search(user_question)

            llm=OpenAI()
            chain=load_qa_chain(llm, chain_type="stuff")
            response=chain.run(input_documents=docs, question=user_question)

            st.write(response)
            



if __name__ == '__main__':
    main()