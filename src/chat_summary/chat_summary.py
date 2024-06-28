from llama_index.core.memory.chat_summary_memory_buffer import ChatSummaryMemoryBuffer
from typing import List
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.utils import get_tokenizer

class ChatSummarizer:
    def __init__(
        self, 
        chat_history:List[ChatMessage],
        llm,
        token_limit:int=800,
        ):
        """
        Initialize the ChatSummarizer with a model, tokenizer, and optional initial chat history.
        
        Parameters:
            model (str): The model name for the LLM.
            tokenizer (Callable): The tokenizer function for the respective model.
            chat_history (List[ChatMessage], optional): Initial chat history. Defaults to None.
            token_limit (int): Token limit for the chat memory buffer. Defaults to 900.
        """
        # self.model = model
        self.tokenizer = get_tokenizer()
        self.llm = llm
        self.chat_history = chat_history
        self.memory = ChatSummaryMemoryBuffer.from_defaults(
            chat_history=chat_history,
            llm=self.llm,
            token_limit=token_limit,
            tokenizer_fn=self.tokenizer,
        )

    def get_history(self):
        """
        Retrieve the current chat history from memory.
        
        Returns:
            List[ChatMessage]: Current chat history.
        """
        return self.memory.get()

    def update_chat_history(self, new_messages):
        """
        Update the chat history with new messages.

        Parameters:
            new_messages (List[ChatMessage]): New messages to be added to the chat history.
        """
        for message in new_messages:
            self.memory.put(message)