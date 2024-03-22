from llama_index.core import StorageContext, load_index_from_storage

async def get_index():
    return load_index_from_storage(StorageContext.from_defaults(persist_dir="/tmp"))

async def set_index():
    storage_context = StorageContext.from_defaults(persist_dir="/tmp")
    storage_context.persist()
