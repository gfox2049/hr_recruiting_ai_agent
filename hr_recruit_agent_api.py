#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install docx2txt --quiet')
get_ipython().system('pip install pypdf --quiet')


# In[2]:


# run behavior
full_app_reload = False


# In[3]:


# NOTE:  process will re-try format errors if resume files have updated


# In[4]:


from datetime import datetime
from dateutil.relativedelta import relativedelta
import uuid
import traceback
import urllib.request
import urllib.error
import os
#from llama_index.core import SimpleDirectoryReader
import boto3
import math
from pypdf import PdfReader
import docx2txt
    
date_after_month = datetime.today()+ relativedelta(months=-4)
print('Today:',datetime.today().strftime("%Y-%m-%dT%H:%M:%SZ"))
print('Months Back:', date_after_month.strftime("%Y-%m-%dT%H:%M:%SZ"))
mon3back = date_after_month.strftime("%Y-%m-%dT%H:%M:%SZ")


# In[5]:


import requests
import base64
import json
import pandas as pd
import numpy as np

API_KEY = ''   # Your Greenhouse API key
#APPLICATION_ID = ""  # Replace with the ID of the application you want to update
#FIELD_NAME_TO_UPDATE = "AI Recommendation"  # Name of the custom field to update
#NEW_FIELD_VALUE = "Pass"  # New value for the custom field
ON_BEHALF_OF = "" # User requesting - (Humberto in example)

# Base URL for Greenhouse Harvest API
BASE_URL = 'https://harvest.greenhouse.io/v1'


# gets all jobs
def get_open_jobs(api_key):

    auth = base64.b64encode(f'{api_key}:'.encode()).decode()
    headers = {
        'Authorization': f'Basic {auth}',
    }

    endpoint = f'{BASE_URL}/jobs'

    payload = {
                #"status": "open",
                "per_page": 500
            }

    response = requests.get(endpoint, headers=headers, params=payload)

    if response.status_code == 200:
        joblist = response.json()
        return joblist
    else:
        print(f'Failed to retrieve. Status code: {response.status_code}, Response: {response.text}')
        return None

def get_job_apps(api_key,page_num,created_after):

    auth = base64.b64encode(f'{api_key}:'.encode()).decode()
    headers = {
        'Authorization': f'Basic {auth}',
    }

    endpoint = f'{BASE_URL}/applications'

    payload = {
                "status": "active",
                "created_after": created_after,
                "per_page": 500,
                "page": page_num,
            }

    response = requests.get(endpoint, headers=headers, params=payload)

    if response.status_code == 200:
        joblist = response.json()
        return joblist
    else:
        print(f'Failed to retrieve. Status code: {response.status_code}, Response: {response.text}')
        return None
    
def list_all_custom_fields(api_key, object_type):
    # Create HTTP basic authentication header
    auth = base64.b64encode(f'{api_key}:'.encode()).decode()
    headers = {
        'Authorization': f'Basic {auth}',
    }

    # Endpoint for retrieving all custom fields for applications
    endpoint = f'{BASE_URL}/custom_fields/{object_type}'

    # Send GET request to list all custom fields
    response = requests.get(endpoint, headers=headers)

    if response.status_code == 200:
        custom_fields = response.json()
        return custom_fields
    else:
        print(f'Failed to retrieve custom fields. Status code: {response.status_code}, Response: {response.text}')
        return None

def update_custom_field(api_key, application_id, object_type, field_id, new_value):
    # API endpoint for updating an application
    url = f"https://harvest.greenhouse.io/v1/{object_type}/{application_id}"

    auth = base64.b64encode(f'{api_key}:'.encode()).decode()
    headers = {
        'Authorization': f'Basic {auth}',
        "Content-Type": "application/json",
        "On-Behalf-Of": ON_BEHALF_OF,
    }

    # The payload with the custom field update
    payload = {
        "custom_fields": [
            {
                "id": field_id,
                "value": new_value
            }
        ]
    }

    # Make the PATCH request to update the custom fields
    response = requests.patch(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        print("Custom field updated successfully.")
    else:
        print(f"Failed to update custom field. Status code: {response.status_code}")
        try:
            print("Response:", response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response content is not JSON:", response.text)


# In[6]:


def bedfunc(question):

    from boto3 import client
    from botocore.config import Config

    max_results_tokens = 3000
    
    config = Config(read_timeout=1000)
    
    bedrock = boto3.client(service_name='bedrock-runtime', 
                          region_name='us-east-1',
                          config=config)

    #bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
    agent = boto3.client(service_name='bedrock-agent-runtime', region_name='us-west-2')

    system_prompt = '''You are Claude, an AI assistant used by Virtasant to be helpful,
            harmless, and honest. Your goal is to provide informative and substantive responses
            to queries while avoiding potential harms.

            You are acting in the role of a Job Recruiting Assistant.

            You are matching candidate resume attributes to requirements of the job description.

            Do not make up any responses that are not present in the context.
            
            '''

    # claude3 body
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_results_tokens,
            "system": system_prompt,
            "temperature": 0, # 0 = minimum randomness
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"\n\nHuman:  {question} \n\nAssistant:"
                        }
                    ]
                }
            ]
        } 
    )
    
    modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0' #'anthropic.claude-3-sonnet-20240229-v1:0'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    
    fin_res = response_body['content'][0]['text']

    return fin_res


# In[7]:


# get all applications in last 3 months (or other date range)

fullapps = []
pnum = 1
len_apps = 1
while len_apps > 0:
    
    appsout = get_job_apps(API_KEY, page_num = pnum, created_after = mon3back)
    len_apps = len(appsout)
    fullapps = fullapps + appsout
    pnum = pnum + 1
    

print('processed rows:',len(fullapps))


# In[8]:


pdapps = pd.json_normalize(fullapps)


# In[9]:


pdapps = pdapps.rename(columns={"id": "application_id"})


# In[10]:


jobz = pdapps['jobs'].str[0].apply(pd.Series)

jobz = jobz.rename(columns={"id": "job_id"})


# In[11]:


pdapps = pd.concat([pdapps, jobz], axis=1)


# In[12]:


jobsout = get_open_jobs(API_KEY)
pdjobs = pd.json_normalize(jobsout)
pdjobs = pdjobs.rename(columns={"id": "job_id"})


# In[13]:


# select only jobs with entered AI fields 
pdjobs = pdjobs[(pdjobs['keyed_custom_fields.ai_pass_threshold.value'] > 0 ) \
& (pdjobs['keyed_custom_fields.ai_possible_threshold.value'] > 0) \
& (len(pdjobs['keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value']) > 0)]
pdjobs.count()


# In[14]:


jobsav = pd.DataFrame()
jobsav.to_csv('jobsav.csv')


# In[15]:


jobsav = pdjobs[['job_id','keyed_custom_fields.ai_pass_threshold.value',
 'keyed_custom_fields.ai_possible_threshold.value',
 'keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value']]


jobsav.to_csv('jobsav.csv')


# In[16]:


# Perform the merge
pdj = pd.merge(pdjobs, jobsav, 
                     on=['job_id'], 
                     how='inner')


# In[17]:


base_columns = [
    'keyed_custom_fields.ai_pass_threshold.value',
    'keyed_custom_fields.ai_possible_threshold.value',
    'keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value'
]
pdj['change_detect'] = False

for base_col in base_columns:
    x_col = f'{base_col}_x'
    y_col = f'{base_col}_y'
    
    # Update 'change_detect' if the values are different
    pdj['change_detect'] |= pdj[x_col] != pdj[y_col]


# In[18]:


column_mappings = {
    'keyed_custom_fields.ai_pass_threshold.value_x': 'keyed_custom_fields.ai_pass_threshold.value',
    'keyed_custom_fields.ai_possible_threshold.value_x': 'keyed_custom_fields.ai_possible_threshold.value',
    'keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value_x': 'keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value'
}

# Rename the columns
pdj = pdj.rename(columns=column_mappings)


# In[19]:


# filter for test set
# remove when done w/ test
#pdjobs = pdjobs[(pdjobs['name'].str.contains('please ignore')) | (pdjobs['name'].str.contains('429'))]
#pdjobs


# In[20]:


pdapps = pdapps.merge(pdjobs, on='job_id', how='inner')


# In[21]:


if full_app_reload == False:
    pdapps = pdapps[pdapps['keyed_custom_fields.ai_recommendation.value'].isnull()]
pdapps.head(5)


# In[22]:


#pdapps = pdapps.sample(20)
rescount = len(pdapps)
print('*** CVs processed:',rescount)


# In[23]:


def get_resume_url(attachments):
    resume = next((item for item in attachments if item['type'] == 'resume'), None)
    return resume['url'] if resume else None

# Apply the function to the 'attachments' column
pdapps['resurl'] = pdapps['attachments'].apply(get_resume_url)


# In[24]:


import os
import shutil

def create_or_recreate_directory(directory_name):
    # Check if the directory already exists
    if os.path.exists(directory_name):
        # If it exists, remove it and its contents
        shutil.rmtree(directory_name)
        print(f"Existing directory '{directory_name}' removed.")
    
    # Create the directory
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' created.")

# Create or recreate 'resdir'
create_or_recreate_directory('resdir')

# Create or recreate 'txtdir'
create_or_recreate_directory('txtdir')


# In[25]:


cset = pdapps


# In[26]:


# subset resume downloads

for index, row in cset.iterrows():

    try:
    
        i = index
    
        fname = 'application_id:' + str(int(cset['application_id'][i]))  + 'candidate_id:' + str(int(cset['candidate_id'][i])) + 'job_id:' + str(int(cset['job_id'][i]))
    
        reslink = cset['resurl'][i]

        response = urllib.request.urlopen(reslink, timeout=4)

        file = open('resdir/' + fname, 'wb')
        file.write(response.read())
        file.close()

        #print('Processed: ',fname)
    
    except:

        f=1
        
        #print('url request exception: ' + str(reslink))

        #print(traceback.format_exc())


# In[27]:


reslist = []
tdir = 'resdir/'
directory = os.fsencode(tdir)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    #print(str(filename))

    try:

        # creating a pdf reader object
        reader = PdfReader(tdir + filename)
        
        # open a text file in write mode
        with open('txtdir/' + filename + '.txt', 'w', encoding='utf-8') as output_file:
            # iterate through all pages
            for page_num in range(len(reader.pages)):
                # create a page object
                page = reader.pages[page_num]
                
                # extract text from page
                text = page.extract_text()
                
                # write the extracted text to the file
                output_file.write(f"Page {page_num + 1}\n")
                output_file.write(text)
                output_file.write('\n\n')  # add some spacing between pages

    except:

        try:
        
            text = docx2txt.process(tdir + filename)
            with open('txtdir/' + filename + '.txt', "w", encoding="utf-8") as txt_file:
                txt_file.write(text)

        except:

            print('non pdf or docx file error: ' + tdir + filename)

            #shutil.copy(tdir + filename, 'txtdir/' + filename + '.txt')

            #print(traceback.format_exc())

# write out list

f=open("subset.txt","w")
f.write(str(reslist))
f.close()


# In[28]:


def func_all(jdesc):

    nlp_query = 'Job Description and Candidate Resume:  \n' + jdesc + ''' 
    
ASK:

Provide the following output: 

CSV row:
    ) 'LLM_CSV_Output'
    ) Candidate First Name
    ) Candidate Last Name
    ) candidate_id
    ) job_id
    ) total possible requirements count
    ) actual matched requirements count
    Column for each of the job description "Must Have" requirements - with 1 indicating a match to candidate or 0 indicating no match:   
    '''

    # Claude 3.5
    agent_response = bedfunc(nlp_query)

    return agent_response


# In[29]:


'''Do not imply experience of candidate that is not explicitly stated in resume.'''

def func_all_short(jdesc):

    nlp_query = 'Job Description and Candidate Resume:  \n' + jdesc + ''' 
    
Instructions:

You are acting as an AI human resources recruiting helper agent.

Ask:

Provide the following output: 

Actual_matches=(actual matched requirements from job to resume as match count)
    
'''

    # Claude 3.5
    agent_response = bedfunc(nlp_query)

    return agent_response


# In[30]:


reslist = []
tdir = 'txtdir/'
directory = os.fsencode(tdir)

cset['cjkey'] = cset['application_id'].astype(int).astype(str) + cset['candidate_id'].astype(int).astype(str) + cset['job_id'].astype(int).astype(str)

app_fs = list_all_custom_fields(api_key = API_KEY, object_type = 'application')
appfd = pd.json_normalize(app_fs)

ai_recon_id = appfd[appfd['name'] == 'AI Recommendation']['id'].values[0]

ai_expl_id = appfd[appfd['name'] == 'AI Explanation']['id'].values[0]

ai_grade_id = appfd[appfd['name'] == 'AI Grade']['id'].values[0]

ai_score_id = appfd[appfd['name'] == 'AI Score']['id'].values[0]


for file in os.listdir(directory):
    filename = os.fsdecode(file)

    fzname = str(filename).replace('application_id:','').replace('candidate_id:','').replace('job_id:','').replace('.txt','')

    # Get app id for exceptions - split the string by 'candidate_id' and take the first part
    first_part = filename.split('candidate_id')[0]

    # Split the first part by ':' and take the second element
    appx_id = first_part.split(':')[1]

    
    try:
    
        with open(tdir + filename) as f:
            s = f.read()
    
        if len(s) > 20:
    
            print(str(filename))
            
            print(fzname + '\n')
    
            subset = cset[cset['cjkey'] == fzname]
    
            app_id = subset['application_id'].values[0]
    
            cz = cset[cset['cjkey'] == fzname]['keyed_custom_fields.ai_guidelines_job_1725019517.7269883.value'].values[0]
    
            passhold = cset[cset['cjkey'] == fzname]['keyed_custom_fields.ai_pass_threshold.value'].values[0]
        
            posshold = cset[cset['cjkey'] == fzname]['keyed_custom_fields.ai_possible_threshold.value'].values[0]
        
            #print('passhold',passhold)
        
            #print('posshold',posshold)
            
            #print('\nRESUME: \n' + s[0:150])
            
            #print('\nJOB DESC: \n' + cz)
    
            llmshort = func_all_short('RESUME: \n' + s + '\n' + 'JOB DESC: \n' + cz)
            print('************')
            #print('\n' + llmshort + '\n')
            print('************')
            
            #llmout = func_all('RESUME: \n' + s + '\n' + 'JOB DESC: \n' + cz)
    
            #print('\n' + llmout + '\n')
    
            #sstring = 'Total_reqs='
            #for line in llmshort.split('\n'):
            #    if sstring in line:
            #        lineout = line.replace(sstring,'')
            #        total_reqs = lineout
    
    
            sstring = 'Actual_matches='
            for line in llmshort.split('\n'):
                if sstring in line:
                    lineout = line.replace(sstring,'')
                    actual_matches = lineout
    
            ai_grade = int(actual_matches) #(int(actual_matches) / int(total_reqs))
    
            ai_score = int(actual_matches)
    
            if ai_score >= passhold:
    
                ai_recon = 'pass'
    
            elif ai_score >= posshold:
    
                ai_recon = 'possible'
    
            else:
    
                ai_recon = 'fail'
    
    
            print('ai_grade',str(ai_grade))
            print('ai_score',str(ai_score))
            print('ai_recon',ai_recon)
    
    
            # update greenhouse
            
            #ai_recon = 'x'
            #llmshort = 'x'
            #ai_grade = 0
            #ai_score = 0
            
    
            
    
            update_custom_field(api_key = API_KEY, application_id = app_id, 
                                object_type = 'applications', 
                                field_id = int(ai_recon_id), 
                                new_value = ai_recon)
    
            update_custom_field(api_key = API_KEY, application_id = app_id, 
                                object_type = 'applications', 
                                field_id = int(ai_expl_id), 
                                new_value = llmshort)
    
            update_custom_field(api_key = API_KEY, application_id = app_id, 
                                object_type = 'applications', 
                                field_id = int(ai_grade_id), 
                                new_value = str(ai_grade))
    
            update_custom_field(api_key = API_KEY, application_id = app_id, 
                                object_type = 'applications', 
                                field_id = int(ai_score_id), 
                                new_value = ai_score)
    
            

    except:

        print('\n*** llm / api exception ***\n')


        update_custom_field(api_key = API_KEY, application_id = appx_id,
                            object_type = 'applications', 
                            field_id = int(ai_expl_id), 
                            new_value = 'ai resume parse - format error - non pdf, word doc or txt file')

        print(traceback.format_exc())
        
        pass


