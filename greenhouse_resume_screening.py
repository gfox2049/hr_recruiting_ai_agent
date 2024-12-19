#!/usr/bin/env python
# coding: utf-8

import os
import json
import base64
import shutil
import requests
import traceback
import urllib.request
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pypdf import PdfReader
import docx2txt
import boto3
from botocore.config import Config

# Configuration
class Config:
    API_KEY = ''  # Greenhouse API key
    ON_BEHALF_OF = ""  # User requesting
    BASE_URL = 'https://harvest.greenhouse.io/v1'
    BEDROCK_REGION = 'us-east-1'
    MAX_RESULTS_TOKENS = 3000
    MONTHS_BACK = 4

class GreenhouseAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.auth_header = self._create_auth_header()

    def _create_auth_header(self):
        auth = base64.b64encode(f'{self.api_key}:'.encode()).decode()
        return {'Authorization': f'Basic {auth}'}

    def get_open_jobs(self):
        endpoint = f'{Config.BASE_URL}/jobs'
        payload = {"per_page": 500}
        response = requests.get(endpoint, headers=self.auth_header, params=payload)
        return self._handle_response(response)

    def get_job_applications(self, page_num, created_after):
        endpoint = f'{Config.BASE_URL}/applications'
        payload = {
            "status": "active",
            "created_after": created_after,
            "per_page": 500,
            "page": page_num
        }
        response = requests.get(endpoint, headers=self.auth_header, params=payload)
        return self._handle_response(response)

    def update_custom_field(self, application_id, field_id, new_value):
        url = f"{Config.BASE_URL}/applications/{application_id}"
        headers = {**self.auth_header, 
                  "Content-Type": "application/json",
                  "On-Behalf-Of": Config.ON_BEHALF_OF}
        
        payload = {
            "custom_fields": [{
                "id": field_id,
                "value": new_value
            }]
        }
        
        response = requests.patch(url, json=payload, headers=headers)
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response):
        if response.status_code == 200:
            return response.json()
        else:
            print(f'API request failed. Status code: {response.status_code}, Response: {response.text}')
            return None

class BedrockAI:
    def __init__(self):
        config = Config(read_timeout=1000)
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=Config.BEDROCK_REGION,
            config=config
        )

    def get_ai_response(self, question):
        system_prompt = '''You are Claude, an AI assistant designed to be helpful,
                harmless, and honest. You are acting in the role of a Job Recruiting Assistant.
                You are matching candidate resume attributes to requirements of the job description.
                Do not make up any responses that are not present in the context.'''

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": Config.MAX_RESULTS_TOKENS,
            "system": system_prompt,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": f"\n\nHuman: {question} \n\nAssistant:"}]
            }]
        })

        response = self.bedrock.invoke_model(
            body=body,
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']

class ResumeProcessor:
    @staticmethod
    def setup_directories():
        for directory in ['resdir', 'txtdir']:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    @staticmethod
    def download_resume(url, filename):
        try:
            response = urllib.request.urlopen(url, timeout=4)
            with open(f'resdir/{filename}', 'wb') as file:
                file.write(response.read())
            return True
        except Exception as e:
            print(f"Error downloading resume: {str(e)}")
            return False

    @staticmethod
    def convert_to_text(filename):
        try:
            reader = PdfReader(f'resdir/{filename}')
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except:
            try:
                return docx2txt.process(f'resdir/{filename}')
            except Exception as e:
                print(f"Error converting file: {str(e)}")
                return None

def main():
    # Initialize classes
    gh_api = GreenhouseAPI(Config.API_KEY)
    bedrock_ai = BedrockAI()
    resume_processor = ResumeProcessor()

    # Setup directories
    resume_processor.setup_directories()

    # Get applications and process them
    # ... (rest of the main processing logic)

if __name__ == "__main__":
    main()
