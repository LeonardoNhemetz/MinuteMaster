import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv


load_dotenv()

def summarize_pdf(pdf_path, groq_chat):
    """
    Summarize the content of a PDF file using the Groq chat model.

    Args:
        pdf_path (str): Path to the PDF file to summarize.
        groq_chat (ChatGroq): The Groq chat object for text summarization.

    Returns:
        str: The summarized content.
    """
    reader = PdfReader(pdf_path)
    text = ""

    # Extract text from all pages in the PDF
    for page in reader.pages:
        text += page.extract_text()

    # Create a prompt to summarize the text
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that summarizes text."),
        HumanMessagePromptTemplate.from_template("Summarize the following text:\n{text}")
    ])

    # Generate the summary
    chain = LLMChain(llm=groq_chat, prompt=prompt)
    summary = chain.predict(text=text)

    return summary

def save_summary_to_pdf(summary, output_path):
    """
    Save the summarized content to a new PDF file.

    Args:
        summary (str): The summarized content.
        output_path (str): Path to save the new PDF file.
    """
    writer = PdfWriter()
    writer.add_page(writer.add_blank_page())

    # Write the summary as a single text page
    writer.pages[0].insert_text(summary)

    with open(output_path, 'wb') as output_pdf:
        writer.write(output_pdf)

def main():
    """
    Main function to summarize all PDF files in the project root.
    """
    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama-3.3-70b-versatile'

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Get list of PDF files in the project root
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the project root.")
        return

    for pdf_file in pdf_files:
        print(f"Summarizing {pdf_file}...")
        summary = summarize_pdf(pdf_file, groq_chat)

        output_file = f"summary_{pdf_file}"
        save_summary_to_pdf(summary, output_file)
        print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    main()
