# from llama_index.vector_stores.azurecosmosmongo.base import VectorStoreQuery
import json
from llama_index.core import PromptTemplate
from typing import Tuple, List
from llama_index.core.llms import ChatMessage, MessageRole
from global_variables import (
    pitch_tester_system_prompt, 
    pitch_tester_easy, 
    pitch_tester_medium, 
    pitch_tester_hard,
    pitch_tester_extreme, 
    pitch_tester_anq_prompt, 
    pitch_tester_anq_easy, 
    pitch_tester_anq_medium, 
    pitch_tester_anq_hard, 
    pitch_tester_anq_extreme, 
    pitch_trainer_easy, 
    pitch_trainer_extreme, 
    pitch_trainer_system_prompt, 
    pitch_trainer_medium, 
    pitch_helper_system_prompt, 
    pitch_trainer_hard,  
    pitch_evaluator_easy, 
    pitch_evaluator_medium, 
    pitch_evaluator_hard, 
    pitch_evaluator_extreme,
    HUGGINGFACEHUB_API_TOKEN
    )

from src.audio_transcription.transcribe import audio_to_text
from src.chat_summary.chat_summary import ChatSummarizer


class MessageProcessor:
        
    first_query = True

    # Creating a template for generating questions based on difficulty
    anq_template = PromptTemplate(
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, {difficulty} please create a complete question:"
    )
    report_template = PromptTemplate(
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, {difficulty} please create a complete evaluation:"
    )

    retrieval_template = PromptTemplate(
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, {difficulty} please create a complete evaluation:"
    )

    helper_template = PromptTemplate(
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please create a complete evaluation:"
    )
    evaluation_template = PromptTemplate(
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please create a complete evaluation:"
    )

    def __init__(
        self, 
        chat_history, 
        retriever,
        llm,
        # transcriber:Transcriber,
        # llama_index_settings:_Settings
        ):

        """
        Initialize the Processor with a ChatSummarizer instance.

        Parameters:
            summarizer (ChatSummarizer): Instance of ChatSummarizer.

        """
        self.chat_history = chat_history
        self.retriever = retriever
        # self.transcriber = transcriber
        # self.llama_index_settings = llama_index_settings
        # 
        self.llm = llm
        self.summarizer = ChatSummarizer(
            chat_history=chat_history,
            llm=llm,
            )
    
    def return_complete_user_message(
        self, 
        message:str, 
        additional_text:str,
        ) -> Tuple[str, List[ChatMessage]]:

        # Transcribe the audio

        # Combine transcribed audio with additional text
        combined_text = message
        if additional_text:
            combined_text += " " + additional_text
            
        current_user_message = ChatMessage(
            content=combined_text,
            )

        self.chat_history.append(current_user_message)
        
        # user_message, history = MessageProcessor.format_user_query(query=current_user_message)
        return self.chat_history[-1].content, self.chat_history

    def format_user_query(self, query) -> Tuple[List[ChatMessage],List]:
        """
        Format and return a list of ChatMessage combining provided query and current chat history.

        Parameters:
            query (str): The user's query in string format.

        Returns:
            List[ChatMessage]: A list of ChatMessage incorporating history and the new user query.
        """
        #format query then add it to the history
        message = [ChatMessage(role=MessageRole.USER, content=query)]
        self.summarizer.update_chat_history([message])
        #return messages with history
        history = self.summarizer.get_history()
        return message, history

    def add_system_response(self, response):
        """
        Updates memory by adding a system response message.

        Parameters:
            response (str): The system's response in string format.
        """
        system_message = ChatMessage(role=MessageRole.SYSTEM, content=response)
        self.summarizer.update_chat_history([system_message])

    # def create_question(self, query_str=[], difficulty="extreme"):
    #     """
    #     Retrieves context based on a query and uses it to ask a question
    #     based on the specified difficulty.

    #     Args:
    #         query_str (str): The query string to retrieve context.
    #         difficulty (str): The level of difficulty to adjust the subsequent prompt.

    #     Returns:
    #         str: A formatted question prompt for user interaction.
    #     """
    #     # Retrieve context based on the query string
    #     context = self.retriever.search(query_str)

    #     if context is None:
    #         return "Unable to retrieve relevant context. Please try a different query."

    #     # Create text prompt based on the retrieved context and difficulty
    #     question = self.anq_template.format(context_str=json.dumps(context), difficulty=difficulty)
        
    #     return question
    
    def query_vector_store(self,query:str) :
        """converts a string into a vector and queris the vector_store
        
        Keyword arguments:
        argument -- description
        Args:
            query (str): the query to embedd
        
        Returns: 
            VectorStoreQueryResult
        """
        pass

    def pitch_helper_handler(self,drop_down_value:str,audio_input:str, additional_text , retriever) ->str:
        """
        Processes audio input through transcription, combines it with any additional text,
        and uses PitchHelper to generate a helpful response in a chat context.

        Args:
            audio_input (str): File path to audio input.
            additional_text (str): Additional text input from the user.

        Returns:
            str: The combined and processed response from the pitch helper system.
        """     
        
        complete_text , history = self.return_complete_user_message(audio_location=audio_input, additional_text=additional_text)

        # embedd_data = self.llama_index_settings.tokenizer(complete_text)
        # padded_embedded_data = extend_array(embedd_data, 1536)

        # vector_store_query = VectorStoreQuery(query_embedding=padded_embedded_data)
        
        vector_store_query = self.query_vector_store(complete_text)

        
        vector_store_query_results = self.retriever.retriever.query(vector_store_query)
        
        context = [i.text for i in vector_store_query_results.nodes]
        
        if context is None:
            raise ValueError("Unable to retrieve relevant context. Please try a different query.")

        # Create text prompt based on the retrieved context and difficulty
        formatted_message = self.retrieval_template.format(context_str=json.dumps(context),difficulty=drop_down_value)
        
        
        self.add_system_response(formatted_message)
        
        messages = self.summarizer.get_history()
        response = self.llm.chat(messages)
        self.add_system_response(response.message.content)
        return response.message.content
    

    def pitch_test_handler(self, drop_down_value:str, audio_input: str, difficulty: str, additional_text: str = "") -> str:
        """
        Handler for pitch training based on the provided audio,
        difficulty level, and additional textual information.

        Args:
            audio_input (str): path to the audio input.
            difficulty (str): difficulty level of training ['easy', 'medium', 'hard', 'extreme'].
            additional_text (str): optional additional text input.

        Returns:
            str: Pitch-related guidance or feedback.
        """

        # Define difficulty to prompt mappings
        initial_difficulty_prompts = {
            'easy': pitch_tester_easy,
            'medium': pitch_tester_medium,
            'hard': pitch_tester_hard,
            'extreme': pitch_tester_extreme
        }
        subsequent_difficulty_prompts = {
            'easy': pitch_tester_anq_easy,
            'medium': pitch_tester_anq_medium,
            'hard': pitch_tester_anq_hard,
            'extreme': pitch_tester_anq_extreme
        }

        context = self.query_vector_store(query=complete_text)
        
        # if context is None:
        #     return "Unable to retrieve relevant context. Please try a different query."

        # Create text prompt based on the retrieved context and difficulty
        # Check if it's the first query and set prompts accordingly
        if first_query:
            # Use initial prompts for the first query
            selected_prompt = pitch_tester_system_prompt + " " + initial_difficulty_prompts.get(difficulty, "")
            self.add_system_response(selected_prompt)
            self.first_query = False
            first_query = False
            complete_text , history = self.return_complete_user_message(
                audio_location=audio_input, 
                additional_text=additional_text
                )
            context = self.retriever.search(query=complete_text)
            formatted_message = self.anq_template.format(context_str=json.dumps(context))
            self.summarizer.update_chat_history(new_messages=formatted_message)
            user_query, chat_messages = self.format_user_query(complete_text)
            response = self.llm.chat(chat_messages)
            self.add_system_response(response.response)
            return response.response

        else:
            selected_prompt = pitch_tester_system_prompt + " " + subsequent_difficulty_prompts.get(difficulty, "")
            self.add_system_response(selected_prompt)
            complete_text , history = self.return_complete_user_message(
                audio_location=audio_input, 
                additional_text=additional_text
                )
            context = self.retriever.search(query=complete_text)
            question_prompt = self.anq_template.format(context_str=json.dumps(context))
            self.add_system_response(question_prompt)
            self.summarizer.update_chat_history(new_messages=formatted_message)
            messages = self.summarizer.get_history()
            # user_query, chat_messages = self.format_user_query(complete_text)
            testquestion = self.llm.chat(messages)
        return testquestion.response

    def pitch_train_handler(self, difficulty: str, audio_location:str,  userquery:str = ""):

        agent_difficulty = {
            'easy': pitch_trainer_easy,
            'medium': pitch_trainer_medium,
            'hard': pitch_trainer_hard,
            'extreme': pitch_trainer_extreme
            }
        selected_prompt = pitch_trainer_system_prompt + " " + agent_difficulty.get(difficulty, "")

        # transcription = audio_to_text(audio_location,api_key=HUGGINGFACEHUB_API_TOKEN)
        transcription = audio_to_text(audio_location)
        
        context = ""

        evaluation = self.evaluation_template.format(context_str=json.dumps(context), difficulty=difficulty)
        self.add_system_response(evaluation)
        complete_text , history = self.return_complete_user_message(
            message=transcription,
            additional_text=userquery,
            )
        context = self.retriever.search(query=complete_text)
        # Process these messages and collect the response
        response = self.llm.chat(history)
        return response.response
    
    def pitch_evaluator(self, difficulty="extreme" , chat_history=[]):
        difficulty = {
            'easy': pitch_evaluator_easy,
            'medium': pitch_evaluator_medium,
            'hard': pitch_evaluator_hard,
            'extreme': pitch_evaluator_extreme
            }
        evaluation = self.report_template.format(difficulty=difficulty.get(difficulty, ""))
        self.add_system_response(evaluation)
        history = self.summarizer.get_history
        response = self.llm.chat(history)
        return response.response