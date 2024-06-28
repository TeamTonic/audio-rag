from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import StorageContext
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
import os
import dotenv

Settings.llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.3", 
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )

# text_splitter = TokenTextSplitter(chunk_size=512)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


documents = SimpleDirectoryReader(os.getenv("DIRECTORY_TO_EMBED")).load_data()


pipeline = IngestionPipeline(
    transformations=[
        # TokenTextSplitter(chunk_size=512),
        SentenceSplitter(chunk_size=512),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    docstore=SimpleDocumentStore(),
)

nodes = pipeline.run(documents=documents)
pipeline.persist("./pipeline_storage")

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)


vector_index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context,
    embed_model=Settings.embed_model,
    )

query_engine = vector_index.as_query_engine(Settings.llm)

if __name__ == '__main__':
    test_output = query_engine.query("hi")

    x=0