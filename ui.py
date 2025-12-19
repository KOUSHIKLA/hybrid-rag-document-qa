import gradio as gr
import yaml
import traceback
from rethinker_retrieval import ask_question

# ---------------------------------------------------------------
# Load ragconfig.yaml
# ---------------------------------------------------------------
with open("ragconfig.yaml", "r") as f:
    cfg = yaml.safe_load(f)

LLM_BASE_URL = cfg["LLM_BASE_URL"]
LLM_API_KEY = cfg["LLM_API_KEY"]
LLM_MODEL = cfg["LLM_MODEL"]

print(f"[INFO] UI using LLM endpoint: {LLM_BASE_URL}")

# ---------------------------------------------------------------
# Query wrapper for UI
# ---------------------------------------------------------------
def run_query(user_query):
    try:
        return ask_question(user_query)
    except Exception as e:
        return "ERROR:\n" + traceback.format_exc()

# ---------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------
with gr.Blocks(title="RAG UI") as demo:

    gr.Markdown(
        """
        #Retrieval-Augmented Generation (RAG) UI  
        Ask questions based on your ingested PDF.
        """
    )

    query = gr.Textbox(
        label="Enter your question",
        placeholder="Example: What is HinToken and HinSent?",
        lines=2
    )

    output = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False
    )

    btn = gr.Button("Submit")
    btn.click(run_query, inputs=query, outputs=output)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
