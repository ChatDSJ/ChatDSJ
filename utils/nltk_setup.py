import nltk
import os
from loguru import logger

def download_nltk_resources():
    """
    Downloads required NLTK resources if they're not already available.
    
    This function should be called during application startup to ensure all
    necessary NLTK data is available.
    """
    # Define the path where NLTK data should be stored
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    # Ensure the directory exists
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the directory to NLTK's search path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    # List of required NLTK resources
    required_resources = [
        'punkt',                   # For tokenization
        'averaged_perceptron_tagger',  # For POS tagging
        'maxent_ne_chunker',       # For named entity recognition
        'words'                    # Required for NER
    ]
    
    # Download each required resource if not already present
    for resource in required_resources:
        try:
            # Check if the resource is already downloaded
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            # Resource not found, so download it
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            logger.success(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Error checking/downloading NLTK resource '{resource}': {e}")

# Run as standalone script to download resources
if __name__ == "__main__":
    download_nltk_resources()