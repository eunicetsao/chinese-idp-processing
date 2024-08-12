"""
PDF Processing and Content Generation Script

This script processes PDF files, extracts text from each page, generates images,
and uses a machine learning model to generate content based on the extracted text
and images. It uses pdfplumber for PDF processing, Jinja2 for template rendering,
and the AWS Bedrock service for content generation.

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
import logging
from typing import List, Dict, Any, Tuple
import boto3
import pdfplumber
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the new package root to the system path
NEW_PACKAGE_ROOT = '/opt/ml/processing/script'
sys.path.insert(0, os.path.abspath(NEW_PACKAGE_ROOT))

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

def find_suitable_image_size(page) -> Tuple[int, int]:
    """
    Determine the suitable image size for a PDF page.
    
    Args:
        page: The PDF page object.
    
    Returns:
        A tuple containing the suitable width and height.
    """
    page_width, page_height = page.width, page.height
    logger.debug(f"PDF page dimensions: {page_width:.2f} x {page_height:.2f}")
    aspect_ratio = page_width / page_height
    
    # Define the accepted aspect ratios and corresponding image sizes
    accepted_ratios: Dict[Tuple[int, int], Tuple[int, int]] = {
        (1, 1): (1092, 1092),
        (3, 4): (951, 1268),
        (2, 3): (896, 1344),
        (9, 16): (819, 1456),
        (1, 2): (784, 1568)
    }
    
    # Find the most suitable aspect ratio
    closest_ratio = min(accepted_ratios.keys(), key=lambda x: abs(aspect_ratio - (x[0] / x[1])))
    suitable_width, suitable_height = accepted_ratios[closest_ratio]
    dpi = min(suitable_width / page_width * 72, suitable_height / page_height * 72)
    logger.debug(f"Calculated DPI: {dpi:.2f}")
    
    # Convert the page to an image
    image = page.to_image(resolution=int(dpi))
    image.save('output_image.png')
    
    return suitable_width, suitable_height

def generate_conversation(
    bedrock_client: boto3.client,
    model_id: str,
    system_text: str,
    input_text_pre: str,
    input_text_post: str,
    input_image: str
) -> Dict:
    """
    Generate a conversation using the Bedrock model.
    
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id: The model ID to use.
        system_text: The system message.
        input_text_pre: The input text before the image.
        input_text_post: The input text after the image.
        input_image: The path to the input image file.
    
    Returns:
        The conversation that the model generated.
    """
    logger.debug(f"Generating message with model {model_id}")
    with open(input_image, "rb") as f:
        image_data = f.read()
    
    inference_config = {"temperature": 0.5}
    additional_model_fields = {"top_k": 3}
    message = {
        "role": "user",
        "content": [
            {"text": input_text_pre},
            {"image": {"format": 'png', "source": {"bytes": image_data}}},
            {"text": input_text_post}
        ]
    }
    
    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": system_text}],
        messages=[message],
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )
    
    logger.debug(f"Model response: {response}")
    return response

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
    
    suitable_width, suitable_height = find_suitable_image_size(page)
    logger.info(f"Suitable image size for page {idx}: {suitable_width}x{suitable_height}")
    
    response = generate_conversation(
        bedrock_client,
        MODEL_ID,
        system_text,
        input_text_pre,
        input_text_post,
        'output_image.png'
    )
    
    return response['output']['message']['content'][0]['text']

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
    results = []
    try:
        with pdfplumber.open(file_path) as pdf_obj:
            for idx, page in enumerate(pdf_obj.pages):
                logger.info(f"Processing page {idx+1} of {file_path}")
                content = process_pdf_page(page, os.path.basename(file_path), idx, templates, bedrock_client)
                results.append(content)
                logger.info(f"Generated content for page {idx+1}")
                logger.debug(f"Generated content: {content}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
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
        logger.info(f"Results saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving results to {output_path}: {str(e)}", exc_info=True)

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
                    logger.info(f"Processing file: {file}")
                    file_path = os.path.join(root, file)
                    results = process_pdf_file(file_path, templates, bedrock_client)
                    save_results(f"{file}_output.txt", results)
                else:
                    logger.info(f"Skipping non-PDF file: {file}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()