# Greenhouse AI Resume Screening Tool

## Overview
This Python script automates the resume screening process for job applications in Greenhouse ATS (Applicant Tracking System). It uses Amazon Bedrock's Claude 3.5 AI model to analyze resumes against job requirements and provides automated scoring and recommendations.

## Features
- Retrieves recent job applications from Greenhouse API
- Downloads and processes resumes in PDF and DOCX formats
- Analyzes resumes against job requirements using Claude 3.5
- Updates Greenhouse with AI recommendations, scores, and explanations
- Supports configurable passing and possible thresholds per job
- Handles multiple file formats and error cases

## Prerequisites
- Python 3.x
- Greenhouse API credentials
- AWS Bedrock access
- Required Python packages:
  - docx2txt
  - pypdf
  - pandas
  - boto3
  - requests
  - numpy

## Configuration
1. Set up environment variables:
   - Greenhouse API key
   - AWS credentials for Bedrock
   - On-behalf-of user ID

2. Job Setup in Greenhouse:
   - Configure custom fields:
     - AI Pass Threshold
     - AI Possible Threshold
     - AI Guidelines
     - AI Recommendation
     - AI Explanation
     - AI Grade
     - AI Score

## Installation
```bash
pip install docx2txt pypdf pandas boto3 requests numpy
```

## Usage
1. Configure API credentials and thresholds
2. Run the script to process recent applications:
```python
python greenhouse_resume_screening.py
```

## Process Flow
1. Retrieves applications from last 3 months
2. Downloads resumes to local directory
3. Converts resumes to text
4. Analyzes content using Claude 3.5
5. Scores matches against job requirements
6. Updates Greenhouse with results

## Output
- Pass/Fail/Possible recommendation
- Matching score
- Detailed explanation of matches
- Grade based on requirements met

## Error Handling
- Handles various resume formats
- Logs format errors
- Retries failed API calls
- Records exceptions in Greenhouse

## Maintenance
- Regular updates for API compatibility
- Threshold adjustments as needed
- Resume format support updates

## Limitations
- Requires specific Greenhouse custom field setup
- PDF and DOCX format support only
- API rate limits apply
