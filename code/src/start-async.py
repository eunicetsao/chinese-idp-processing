"""
PDF Processing and Content Generation Script

This script processes PDF files, extracts text from each page, and generates content
based on the extracted text using a machine learning model. It uses pdfplumber for
PDF processing, Jinja2 for template rendering, and the AWS Bedrock service for
content generation.

The script walks through a specified directory, processes all PDF files it finds,
and saves the generated content for each file in an output directory.

Constants:
    TEMPLATE_DIR (str): Directory containing Jinja2 templates
    DATASET_DIR (str): Input directory containing PDF files to process
    OUTPUT_DIR (str): Output directory for saving generated content
    MODEL_ID (str): ID of the machine learning model to use
    REGION_NAME (str): AWS region name for the Bedrock service

Usage:
    Run this script directly: python script_name.py
"""

import os
import sys
from typing import List, Dict, Any
import boto3
import asyncio
import pdfplumber
from jinja2 import Environment, FileSystemLoader

# Add the new package root to the system path
NEW_PACKAGE_ROOT = '/opt/ml/processing/script'
sys.path.insert(0, os.path.abspath(NEW_PACKAGE_ROOT))

from utils import find_suitable_image_size, generate_conversation

# Constants
TEMPLATE_DIR = '/opt/ml/processing'
DATASET_DIR = '/opt/ml/processing/input'
OUTPUT_DIR = '/opt/ml/processing/output'
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
REGION_NAME = 'us-west-2'


def setup_jinja_environment() -> Environment:
    """
    Set up and return the Jinja environment.

    Returns:
        Environment: Configured Jinja environment object.
    """
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR))

def load_templates(env: Environment) -> Dict[str, Any]:
    """
    Load and return the Jinja templates.

    Args:
        env (Environment): Jinja environment object.

    Returns:
        Dict[str, Any]: Dictionary containing loaded Jinja templates.
    """
    return {
        'system_prompt': env.get_template('script/system_prompt_template.jinja'),
        'user_prompt_pre': env.get_template('script/user_prompt_pre_template.jinja'),
        'user_prompt_post': env.get_template('script/user_prompt_post_template.jinja')
    }

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content + '\n')

def process_pdf_page(page: Any, file: str, idx: int, templates: Dict[str, Any], bedrock_client: Any) -> str:
    """
    Process a single PDF page and return the generated content.

    Args:
        page (Any): PDF page object from pdfplumber.
        file (str): Name of the PDF file being processed.
        idx (int): Index of the current page.
        templates (Dict[str, Any]): Dictionary of loaded Jinja templates.
        bedrock_client (Any): Boto3 client for the Bedrock service.

    Returns:
        str: Generated content for the page.
    """
    pdf_text = page.extract_text(layout=True)
    system_text = templates['system_prompt'].render()
    input_text_pre = templates['user_prompt_pre'].render(PDF_TEXT=pdf_text, FILENAME=file, PAGE_NUMBER=idx)
    input_text_post = templates['user_prompt_post'].render()
    
    suitable_image_size = find_suitable_image_size(page)
    input_image = './output_image.png'
    
    response = generate_conversation(
        bedrock_client,
        MODEL_ID,
        system_text,
        input_text_pre,
        input_text_post,
        input_image
    )
    
    filename = OUTPUT_DIR + "/idp_output.txt"
    content = response['output']['message']['content'][0]['text']
    write_to_file(filename, content)
    
    return None

async def async_calls_on_model(object):
    return await asyncio.to_thread(process_pdf_page, object)

async def parallel_calls(objects):
    start_time = time.time()    
    await asyncio.gather(*[async_calls_on_model(obj) for obj in objects[:10]])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nAll tasks completed in {:.2f} seconds".format(elapsed_time))
    return None

def process_pdf_file(file_path: str, templates: Dict[str, Any], bedrock_client: Any) -> List[str]:
    """
    Process a single PDF file and return a list of generated content for each page.

    Args:
        file_path (str): Path to the PDF file.
        templates (Dict[str, Any]): Dictionary of loaded Jinja templates.
        bedrock_client (Any): Boto3 client for the Bedrock service.

    Returns:
        List[str]: List of generated content for each page in the PDF.
    """
    objects = []
    results = []
    try:
        with pdfplumber.open(file_path) as pdf_obj:
            for idx, page in enumerate(pdf_obj.pages):
                file = os.path.basename(file_path)
                obj = {"page": page, "file": file, "idx": idx, "templates": templates, "bedrock_client": bedrock_client}
                objects.append(obj)   
        loop = asyncio.get_event_loop()
        loop.create_task(parallel_calls(objects))
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    return results



def save_results(file_name: str, results: List[str]) -> None:
    """
    Save the results to a file.

    Args:
        file_name (str): Name of the file to save results to.
        results (List[str]): List of generated content to save.
    """
    output_path = os.path.join(OUTPUT_DIR, file_name)
    try:
        with open(output_path, "w") as f:
            for result in results:
                f.write(result + "\n")
    except IOError as e:
        print(f"Error saving results to {output_path}: {str(e)}")


def main() -> None:
    """
    Main function to orchestrate the PDF processing workflow.

    This function sets up the Jinja environment, loads templates, initializes the
    Bedrock client, and processes all PDF files in the specified input directory.
    Generated content is saved to the output directory.
    """
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
        env = setup_jinja_environment()
        templates = load_templates(env)
   
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                if file.lower().endswith('.pdf'):
                    print(f"Processing: {file}")
                    file_path = os.path.join(root, file)
                    results = process_pdf_file(file_path, templates, bedrock_client)
                    save_results(f"{file}_output.txt", results)
                else:
                    print(f"Skipping non-PDF file: {file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()