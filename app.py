import os
import sys
import argparse
import gradio as gr
import time
from dotenv import load_dotenv
load_dotenv()

# Force Hugging Face progress bars to print directly to the Kaggle terminal console
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from insert_data.build_chromadb import BuildChromaDB
from custom_router.router import SemanticRouter
from custom_router.routes import research_route, chitchat_route
from llm.LLM import LLM
from rag.core import RAG

parser = argparse.ArgumentParser(description="Run the RAG Assistant Interface")
parser.add_argument("-k", "--top_k", type=int, default=5, 
                    help="Number of top reranked chunks to feed to the LLM context window")
# Using a default of 5 for LLM injection
args = parser.parse_args()

##################### CONFIGURATIONS #####################################
RAG_PROMPT = """You are an assistant that specialises in reading and analysing research papers, 
based on the given question and the context chunks, 
Answer the question without any information outside the sources you are given. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

current_query = {"value": None}

def get_env(key, default=None):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key)
    except Exception:
        return os.getenv(key, default)
    
builder = BuildChromaDB(
    collection_name="project1",
    model_cache_dir=get_env("MODEL_CACHE_DIR", "./model_cache"),
    api_key=get_env("CHROMA_API_KEY"),
    tenant=get_env("CHROMA_TENANT"),
    database=get_env("CHROMA_DATABASE")
)
router = SemanticRouter(
    routes=[research_route, chitchat_route]
)
llm = LLM(api_key=get_env("GEMINI_API_KEY"),
          model_name="gemini-2.5-flash")
rag = RAG(chromadb_api_key=get_env("CHROMA_API_KEY"),
          tenant_key=get_env("CHROMA_TENANT"),
          db_key=get_env("CHROMA_DATABASE")
          )

##################### HELPER FUNCTIONS #####################################
def index_on_upload(file):
    if file is None:
        return "No file uploaded."
    if not file.name.lower().endswith(".pdf"):
        return f"Rejected: {file.name} is not a PDF."
    with open(file.name, "rb") as f:
        if f.read(5) != b"%PDF-":
            return "Rejected: invalid PDF content."

    ids = builder.build_(file.name)
    return f"Indexed {len(ids)} chunks into 'project1'."

def chat_interface(query, history, selected_model, context_limit):
    if not query.strip():
        yield history, ""
        return
    
    current_query["value"] = query
    llm.model_name = selected_model
    
    # Immediately append the user's question and show the initial status message
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "Routing query intent..."}
    ]
    yield history, ""
    
    # Execute semantic routing
    route = router.guide(query)
    
    if route == "chitchat":
        history[-1]["content"] = "Thinking..."
        yield history, ""
        
        bot_response = llm.create_content(query)
        history[-1]["content"] = bot_response
        yield history, ""
        
    else:
        history[-1]["content"] = "Retrieving and reranking contextual document chunks..."
        yield history, ""
        
        results = rag.vector_search(query, limit=int(context_limit))
        
        time.sleep(0.3) 
        
        history[-1]["content"] = "Generating formal response text..."
        yield history, ""
        
        context = "\n\n".join(r["chunk"] for r in results)
        prompt = RAG_PROMPT.format(context=context, question=query)
        
        bot_response = llm.create_content(prompt)
        
        history[-1]["content"] = bot_response
        yield history, ""

def get_current_query():
    return current_query["value"] or "No query saved yet."


with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# Document RAG Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload PDF")
            file_input = gr.File(label="Document", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            file_input.change(index_on_upload, inputs=file_input, outputs=upload_status)
            
            model_dropdown = gr.Dropdown(
                choices=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash"],
                value="gemini-2.5-flash",
                label="Gemini Model Engine",
                interactive=True
            )
            
            k_slider = gr.Slider(
                minimum=1, 
                maximum=20, 
                value=args.top_k, 
                step=1, 
                label="LLM Context Chunk Limit (Top-N)",
                interactive=True
            )

        with gr.Column(scale=2):
            gr.Markdown("### Ask a question")
            chatbot = gr.Chatbot(height=400, type="messages") 
            query_box = gr.Textbox(placeholder="Type your question and press Enter", show_label=False)
            
            query_box.submit(
                chat_interface, 
                inputs=[query_box, chatbot, model_dropdown, k_slider], 
                outputs=[chatbot, query_box]
            )

    with gr.Accordion("Debug: current_query value", open=False):
        debug_btn = gr.Button("Refresh")
        debug_output = gr.Textbox(label='current_query["value"]')
        debug_btn.click(get_current_query, outputs=debug_output)

if __name__=="__main__":
    demo.launch(share=True)