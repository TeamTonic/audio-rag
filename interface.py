import asyncio
from global_variables import (
    title,
    description,
    article_trainer,
    article_helper,
    article_tester,
    )
import gradio as gr
from src.message_processor.message_processor import MessageProcessor
from llama_index.core.base.llms.types import (
    ChatMessage,
    )
from llama_index.core.memory.chat_summary_memory_buffer import  ChatSummaryMemoryBuffer
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from  src.documents_handler.load_documents import query_engine
from src.documents_handler.load_documents import vector_index
import os

llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.3", 
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

    
vector_index_retriever = vector_index.as_retriever()

# Init Chat History & Mempory
chat_history = [ChatMessage()]
chat_memory = ChatSummaryMemoryBuffer(
    token_limit=900,
    chat_history=chat_history,
    )


message_processor = MessageProcessor(
    retriever=vector_index_retriever,
    chat_history=chat_history,
    llm=llm,
)

demo = gr.Blocks()

pitch_trainer = gr.Interface(
fn=message_processor.pitch_train_handler,
inputs=[
    gr.Dropdown(type="value", value="hard", choices=["easy", "medium", "hard", "extreme"]),
    gr.Audio(label="Use Your Microphone For Best Results" , type= "filepath"),
    gr.Textbox(label="Add Additional Information Via Text Here", ),
],
outputs=[
    gr.Textbox(label="Tonic Pitch Trainer"),
],
allow_flagging="never",
title=title,
description=description,
article=article_trainer,
)

with demo:
    gr.TabbedInterface([
        # pitch_helper, 
        # pitch_tester, 
        pitch_trainer
        ], ["Tonic Pitch Assistant", "Test Your Pitching", "Train For Your Pitch"])
demo.queue(max_size=5)
demo.launch(server_name="localhost", show_api=False)

# if __name__ == '__main__':
# asyncio.run(main())