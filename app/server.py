import os, time, openai, asyncio, uvicorn
from fastapi import FastAPI, Request, HTTPException
from typing import AsyncIterable, Awaitable
from decouple import config
from starlette.responses import JSONResponse
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from app.utilities import connect, format_docs
from decorators.jwt import jwt_token
from decorators.key import x_app_key
from decorators.teams import x_super_team
from decouple import config

app = FastAPI()

@app.post('/bots/create')
@x_super_team
@x_app_key
@jwt_token
async def create_bots(request: Request):
        try:
            data = await request.form()

            if not data.get('bot_name'):
                raise HTTPException(status_code = 400, detail = "Missing 'bot_name' parameter!")
            if not data.get('openai_key'):
                raise HTTPException(status_code = 400, detail = "Missing 'openai_key' parameter!")
            if not data.get('chat_history'):
                raise HTTPException(status_code = 400, detail = "Missing 'chat_history' parameter!")
            
            name, openai_key, chat_history = data.get('bot_name'), data.get('openai_key'), data.get('chat_history')

            if not chat_history == 'enable' and not chat_history == 'disable':
                raise HTTPException(status_code = 400, detail = "Invalid 'chat_history' parameter!")

            company_id = request.headers.get('x-super-team')
            user = request.state.current_user

            db = connect()
            bots_collections = db['bots']

            latest_bot = bots_collections.find_one(sort=[("created_date", -1)])
            if latest_bot:
                bot_id = f"{int(latest_bot['bot_id']) + 1}"
            else: 
                bot_id = '1'
                
            bots_documents = bots_collections.find({"company_id": company_id, "bot_name": name, "is_active": 1})
            bots_records = list(bots_documents) 

            for record in bots_records:
                if record['bot_name'] == name:
                    return JSONResponse(f"Name '{name}' already exists.")

            client = openai.OpenAI(api_key = openai_key)
            try:
                client.models.list()
            except openai.AuthenticationError:
                raise HTTPException(status_code = 400, detail = f"Invalid OpenAI key!")
            
            if not os.path.exists(f'library/{company_id}/{bot_id}/documents'):
                os.makedirs(f'library/{company_id}/{bot_id}/documents')  
                os.makedirs(f'library/{company_id}/{bot_id}/embeddings') 

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")

            document = {'company_id': company_id, 'bot_id': bot_id, "bot_name": name, 'openai_key': openai_key, "chat_history": chat_history, 'is_active': 1, 'created_date': date_time, 'modified_date': date_time, 'created_by': user, 'modified_by': user}
           
            bots_collections.insert_one(document)

            return JSONResponse(f"'{name}' bot initialised successfully!")

        except HTTPException as e:
                raise e
        except Exception as e:
            raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/bots/get')
@x_super_team
@x_app_key
@jwt_token
async def get_bots(request: Request):
        try:
            company_id = request.headers.get('x-super-team')

            db = connect()
            bots_collections = db['bots']
                
            bots_documents = bots_collections.find({"company_id": company_id, "is_active": 1})
            bots_records = list(bots_documents) 

            if bots_records:
                results = []
                for record in bots_records:
                    temp = {'bot_id': record['bot_id'], 'bot_name': record['bot_name'], 'openai_key': record['openai_key'], 'chat_history': record['chat_history']}
                    results.append(temp)
            
                return JSONResponse(results)
            else:
                raise HTTPException(status_code = 400, detail = f"No available bots found for company id {company_id}!")

        except HTTPException as e:
                raise e
        except Exception as e:
            raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
        
@app.post('/bots/disable')
@x_super_team
@x_app_key
@jwt_token
async def disable_bots(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")

        bot_id = data.get('bot_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()
    
        bots_collections = db['bots']

        bots_documents = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

        if bots_documents:                
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")

            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"is_active": 0}})
            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"modified_date": date_time}})
            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"modified_by": user}})

            return JSONResponse(f'Bot ID {bot_id} disabled successfully!')
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/bots/enable')
@x_super_team
@x_app_key
@jwt_token
async def enable_bots(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")

        bot_id = data.get('bot_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()
    
        bots_collections = db['bots']

        bots_documents = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 0})

        if bots_documents:                
            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")

            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"is_active": 1}})
            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"modified_date": date_time}})
            bots_collections.update_one({"_id": bots_documents["_id"]}, {"$set": {"modified_by": user}})

            return JSONResponse(f'Bot ID {bot_id} enabled successfully!')
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/documents/create')
@x_super_team
@x_app_key
@jwt_token
async def upload_documents(request: Request):
    try:
        data = await request.form()

        print(data)

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('documents'):
            raise HTTPException(status_code = 400, detail = "Missing 'documents' parameter!")

        bot_id, files = data.get('bot_id'), data.getlist('documents')
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code = 400, detail = "Only PDFs allowed!")

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()

        bots_collections = db['bots']
        library_collections = db['library']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            for file in files:                 
                    
                library_documents = library_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1, "file_name": file.filename})
                library_records = list(library_documents) 

                if library_records and library_records[0]['file_name'] == file.filename:
                    raise HTTPException(status_code = 400, detail = f"Document '{file.filename}' already exist!")
                
                latest_document = library_collections.find_one({"company_id": company_id, "bot_id": bot_id}, sort=[("created_date", -1)])
                if latest_document:
                    document_id = f"{int(latest_document['document_id']) + 1}"
                else: 
                    document_id = '1'

                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")

                document = {'company_id': company_id, 'bot_id': bot_id, 'document_id': document_id, 'file_name': file.filename, 'is_active': 1, 'created_date': date_time, 'modified_date': date_time, 'created_by': user, 'modified_by': user}
                library_collections.insert_one(document)

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})

                file_path = f"library/{company_id}/{bot_id}/documents/{file.filename}"
                with open(file_path, "wb") as buffer:
                    buffer.write(await file.read())  

                time.sleep(1)

            return JSONResponse(f"Document(s) uploaded successfully!") 
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/documents/get')
@x_super_team
@x_app_key
@jwt_token
async def get_documents(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")

        bot_id = data.get('bot_id')

        company_id = request.headers.get('x-super-team')

        db = connect()

        bots_collections = db['bots']
        library_collections = db['library']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            library_documents = library_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
            library_records = list(library_documents) 
            
            if library_records:
                results = []
                for record in library_records:
                    temp = {'document_id': record['document_id'], 'file_name': record['file_name']}
                    results.append(temp)
                            
                return JSONResponse(results)
            else:
                raise HTTPException(status_code = 400, detail = f"No available documents found for bot id {bot_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/documents/disable')
@x_super_team
@x_app_key
@jwt_token
async def disable_documents(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('document_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'document_id' parameter!")

        bot_id, document_id = data.get('bot_id'), data.get('document_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()
    
        bots_collections = db['bots']
        library_collections = db['library']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            library_document = library_collections.find_one({"company_id": company_id, "bot_id": bot_id, "document_id": document_id, "is_active": 1})
            
            if library_document:
                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")

                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"is_active": 0}})
                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"modified_date": date_time}})
                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"modified_by": user}})

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})

                return JSONResponse(f'Document ID {document_id} disabled successfully!')
            else:
                raise HTTPException(status_code = 400, detail = f"No available document found for id {document_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/documents/enable')
@x_super_team
@x_app_key
@jwt_token
async def enable_documents(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('document_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'document_id' parameter!")

        bot_id, document_id = data.get('bot_id'), data.get('document_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()
    
        bots_collections = db['bots']
        library_collections = db['library']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            library_document = library_collections.find_one({"company_id": company_id, "bot_id": bot_id, "document_id": document_id, "is_active": 0})
            
            if library_document:
                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")

                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"is_active": 1}})
                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"modified_date": date_time}})
                library_collections.update_one({"_id": library_document["_id"]}, {"$set": {"modified_by": user}})

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})

                return JSONResponse(f'Document ID {document_id} enabled successfully!')
            else:
                raise HTTPException(status_code = 400, detail = f"No available document found for id {document_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
            raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/embeddings/create')
@x_super_team
@x_app_key
@jwt_token
async def create_embeddings(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('embeddings_name'):
            raise HTTPException(status_code = 400, detail = "Missing 'embeddings_name' parameter!")
        if not data.get('extract_images'):
            raise HTTPException(status_code = 400, detail = "Missing 'extract_images' parameter!")

        bot_id, embeddings_name, extract_images = data.get('bot_id'), data.get('embeddings_name'), data.get('extract_images')

        if not extract_images == 'enable' and not extract_images == 'disable':
            raise HTTPException(status_code = 400, detail = f"Invalid 'extract_images' parameter!")
        
        if extract_images == 'enable':
            extract_images = True
        elif extract_images == 'disable':
            extract_images = False
        
        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()

        bots_collections = db['bots']
        documents_collections = db['library']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        embeddings_documents = embeddings_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1, "embeddings_name": embeddings_name})
        embeddings_records = list(embeddings_documents) 

        if embeddings_records:
            raise HTTPException(status_code = 400, detail = f"Embeddings name '{embeddings_name}' already exist!")

        if bots_records:
            library_documents = documents_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
            library_records = list(library_documents) 

            if library_records:
                data = []
                for record in library_records:
                    loader = PyMuPDFLoader(f"library/{company_id}/{bot_id}/documents/{record['file_name']}", extract_images = extract_images)
                    data.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ".", ",", ""],
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len,
                    is_separator_regex=False,
                )
                
                chunks = text_splitter.split_documents(data)

                embeddings = OpenAIEmbeddings(model = config("TEXT_EMBEDDING_MODEL"), openai_api_key = bots_records[0]['openai_key'])
                vector_store = FAISS.from_documents(chunks, embeddings)

                vector_store.save_local(f"library/{company_id}/{bot_id}/embeddings/{embeddings_name}")
                
                latest_embeddings = embeddings_collections.find_one({"company_id": company_id, "bot_id": bot_id}, sort=[("created_date", -1)])
                if latest_embeddings:
                    embeddings_id = f"{int(latest_embeddings['embeddings_id']) + 1}"
                else: 
                    embeddings_id = '1'

                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        
                document = {'company_id': company_id, 'bot_id': bot_id, "embeddings_id": embeddings_id, "embeddings_name": embeddings_name, 'is_active': 1, 'created_date': date_time, 'modified_date': date_time, 'created_by': user, 'modified_by': user}
                embeddings_collections.insert_one(document)   

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})
                                
                return JSONResponse('Embedding created successfully.')
            else:
                raise HTTPException(status_code = 400, detail = f"No available documents found for bot id {bot_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/embeddings/get')
@x_super_team
@x_app_key
@jwt_token
async def get_embeddings(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")

        bot_id = data.get('bot_id')

        company_id = request.headers.get('x-super-team')

        db = connect()

        bots_collections = db['bots']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            embeddings_documents = embeddings_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
            embeddings_records = list(embeddings_documents) 

            if embeddings_records:
                results = []
                for record in embeddings_records:
                    temp = {'embeddings_id': record['embeddings_id'], 'embeddings_name': record['embeddings_name']}
                    results.append(temp)
                            
                return JSONResponse(results)
            else:
                raise HTTPException(status_code = 400, detail = f"No available embeddings found for bot id {bot_id}!")
            
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/embeddings/disable')
@x_super_team
@x_app_key
@jwt_token
async def disable_embeddings(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('embeddings_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'embeddings_id' parameter!")

        bot_id, embeddings_id = data.get('bot_id'), data.get('embeddings_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()

        bots_collections = db['bots']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            embeddings_document = embeddings_collections.find_one({"company_id": company_id, "bot_id": bot_id, "embeddings_id": embeddings_id, "is_active": 1})
            
            if embeddings_document:
                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")

                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"is_active": 0}})
                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"modified_date": date_time}})
                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"modified_by": user}})

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})

                return JSONResponse(f'Embedding ID {embeddings_id} disabled successfully!')
            else:
                raise HTTPException(status_code = 400, detail = f"No available embeddings found for id {embeddings_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/embeddings/enable')
@x_super_team
@x_app_key
@jwt_token
async def enable_embeddings(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('embeddings_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'embeddings_id' parameter!")

        bot_id, embeddings_id = data.get('bot_id'), data.get('embeddings_id')

        company_id = request.headers.get('x-super-team')
        user = request.state.current_user

        db = connect()

        bots_collections = db['bots']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            embeddings_document = embeddings_collections.find_one({"company_id": company_id, "bot_id": bot_id, "embeddings_id": embeddings_id, "is_active": 0})
            
            if embeddings_document:
                now = datetime.now()
                date_time = now.strftime("%d/%m/%Y %H:%M:%S")

                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"is_active": 1}})
                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"modified_date": date_time}})
                embeddings_collections.update_one({"_id": embeddings_document["_id"]}, {"$set": {"modified_by": user}})

                bots_document = bots_collections.find_one({"company_id": company_id, "bot_id": bot_id, "is_active": 1})

                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_date": date_time}})
                bots_collections.update_one({"_id": bots_document["_id"]}, {"$set": {"modified_by": user}})

                return JSONResponse(f'Embedding ID {embeddings_id} enabled successfully!')
            else:
                raise HTTPException(status_code = 400, detail = f"No available embeddings found for id {embeddings_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
    
@app.post('/embeddings/response')
@x_super_team
@x_app_key
@jwt_token
async def response_embeddings(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('k'):
            raise HTTPException(status_code = 400, detail = "Missing 'k' parameter!")
        if not data.get('text'):
            raise HTTPException(status_code = 400, detail = "Missing 'text' parameter!")

        bot_id, k, text = data.get('bot_id'), data.get('k'), data.get('text')

        company_id = request.headers.get('x-super-team')

        db = connect()

        bots_collections = db['bots']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            embeddings_documents = embeddings_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
            embeddings_records = list(embeddings_documents) 

            if embeddings_records:
                embeddings = OpenAIEmbeddings(model = config("TEXT_EMBEDDING_MODEL"), openai_api_key = bots_records[0]['openai_key'])
                path = f"library/{company_id}/{bot_id}/embeddings/{embeddings_records[0]['embeddings_name']}"

                vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization = True)

                retriever = vectorstore.as_retriever(search_kwargs={"k": int(k)})

                documents = retriever.invoke(text)
                documents = format_docs(documents)
                            
                return JSONResponse(documents)
            else:
                raise HTTPException(status_code = 400, detail = f"No available embeddings found for bot id {bot_id}!")
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

@app.post('/chain/stream')
@x_super_team
@x_app_key
@jwt_token
async def stream_chain(request: Request):
    try:
        data = await request.form()

        if not data.get('bot_id'):
            raise HTTPException(status_code = 400, detail = "Missing 'bot_id' parameter!")
        if not data.get('text'):
            raise HTTPException(status_code = 400, detail = "Missing 'text' parameter!")

        bot_id, text = data.get('bot_id'), data.get('text')

        company_id = request.headers.get('x-super-team')

        db = connect()

        bots_collections = db['bots']
        embeddings_collections = db['embeddings']

        bots_documents = bots_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
        bots_records = list(bots_documents) 

        if bots_records:
            embeddings_documents = embeddings_collections.find({"company_id": company_id, "bot_id": bot_id, "is_active": 1})
            embeddings_records = list(embeddings_documents) 

            if embeddings_records:
                embeddings = OpenAIEmbeddings(model = config("TEXT_EMBEDDING_MODEL"), openai_api_key = bots_records[0]['openai_key'])
                path = f"library/{company_id}/{bot_id}/embeddings/{embeddings_records[0]['embeddings_name']}"

                vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization = True)

                if len(embeddings_records) > 1:
                    for record in embeddings_records[1:]:
                        path = f"library/{company_id}/{bot_id}/embeddings/{record['embeddings_name']}"

                        temp = FAISS.load_local(path, embeddings, allow_dangerous_deserialization = True)
                        vectorstore.merge_from(temp)

                retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
                chat_history = bots_records[0]['chat_history']

                if chat_history == 'enable':
                    if not data.get('session_id'):
                        raise HTTPException(status_code = 400, detail = "Missing 'session_id' parameter!")

                    session_id = data.get('session_id')
                    return StreamingResponse(response_with_history(text, retriever, session_id), media_type = 'text/event-stream')
                
                if chat_history == 'disable':
                    return StreamingResponse(response_without_history(text, retriever), media_type = 'text/event-stream')
            else:
                raise HTTPException(status_code = 400, detail = f"No available embeddings found for bot id {bot_id}!")  
        else:
            raise HTTPException(status_code = 400, detail = f"Bot ID {bot_id} doesn't exist!")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")

async def response_with_history(text, retriever, session_id) -> AsyncIterable[str]:
    
    callback = AsyncIteratorCallbackHandler()

    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0, api_key = config("OPENAI_API_KEY"))

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which"
        "might reference context in the chat history, formulate a standalone"
        "question which can be understood without the chat history." 
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm_out = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0, api_key = config("OPENAI_API_KEY"), streaming=True, verbose=True, callbacks=[callback])

    question_answer_chain = create_stuff_documents_chain(llm_out, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    host = config("DATABASE_HOST")
    username = config("DATABASE_USERNAME")
    password = config("DATABASE_PASSWORD")
    database = config("DATABASE_NAME")

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=f"mongodb://{host}:27017",
            database_name=f"{database}",
            collection_name="history",
        ),
        input_messages_key="input",
        history_messages_key="chat_history",        
        output_messages_key="answer",
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
        finally:
            event.set()

    task = asyncio.create_task(wrap_done(
        chain_with_history.ainvoke({'input': text}, config={
        "configurable": {"session_id": session_id}
    }),
        callback.done),
    )

    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task

async def response_without_history(text, retriever) -> AsyncIterable[str]:
    
    callback = AsyncIteratorCallbackHandler()

    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0, api_key = config("OPENAI_API_KEY"), streaming=True, verbose=True, callbacks=[callback])

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        try:
            await fn
        except Exception as e:
            raise HTTPException(status_code = 500, detail = f"An error occurred: {str(e)}")
        finally:
            event.set()

    task = asyncio.create_task(wrap_done(
        rag_chain.ainvoke(text),
        callback.done),
    )

    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
