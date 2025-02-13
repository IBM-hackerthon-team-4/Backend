import os
from fastapi import FastAPI, Form
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class PromptMessage(BaseModel):
    prompt: str

class Message(BaseModel):
    text: str

def create_llm(api_key, api_url, project_id):
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }
    
    credentials = Credentials(
        url=api_url,
        api_key=api_key
    )
    
    model_id = "mistralai/mistral-large"
    #model_id = "meta-llama/llama-3-3-70b-instruct" 
    llm = ModelInference(
        model_id=model_id,
        params = parameters,
        credentials=credentials,
        project_id=project_id
    )
    return llm

api_key = ""
api_url = ""
project_id = ""

model = create_llm(api_key, api_url, project_id)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/processing", response_model=Message)
async def watsonx_ai_api(prompt_message: PromptMessage):
    try:
        response = model.generate(prompt=prompt_message.prompt)['results'][0]['generated_text'].strip()
        print(response)
        msg = {"text": response}
        return msg
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

