import os
import base64
from dotenv import load_dotenv
from IPython.display import display, Image
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DATA_DIR = os.path.join(BASE_DIR, "sim_data")
FIGURE_DIR = os.path.join(SIM_DATA_DIR, "figures")

# Initialize the Azure OpenAI client
API_KEY = os.getenv("AZURE_API_KEY")
DEPLOYMENT_ENDPOINT = os.getenv("AZURE_DEPLOYMENT_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("GPT4o_DEPLOYMENT_NAME")
API_VERSION = "2023-03-15-preview"

client = AzureOpenAI(
    azure_endpoint=DEPLOYMENT_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_images(image_filenames, added_context):
    """
    Analyzes one or more images using GPT-4.

    Parameters:
    image_filenames (list): A list of image filenames to be analyzed.
    added_context (str): Additional context to provide to the LLM, e.g. parameters of simulations.

    Returns:
    str: The response message from GPT-4.
    """
    
    # Ensure image_filenames is a list even if a single filename is provided
    if isinstance(image_filenames, str):
        image_filenames = [image_filenames]

    # Full paths to images
    image_paths = [os.path.join(FIGURE_DIR, filename) for filename in image_filenames]

    # Encode images and prepare them for the prompt
    base64_images = [encode_image(image_path) for image_path in image_paths]

    # Display images for visual reference
    for image_path in image_paths:
        display(Image(image_path))

        
    sys_prompt = """
    You are an engineering assistant specializing in water-related projects, focusing on numerical simulations and equipped to analyze images of 
    2d simulation results. 
    You will be asked to analyze images of water-related simulations and provide a brief, concise description of the results.
    Describe features such as minima and maxima, trends, and patterns are of interest.
    Expect to contrast the image with other images e.g. from other timesteps of the simulation or other simulations.

    Exemplary descriptions:
    * "Water levels range from 0 to 5 meters. A wave is located at x=100m and skewed to the right."
    * "The wave propagation speeds are similar in both, but the wave dissipates faster in the second simulation."
    """

    if added_context is not None:
        sys_prompt += added_context

    messages = [
        {"role": "system", "content": sys_prompt},
    ]

    for base64_image in base64_images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        })

    # Call AI
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
    )

    
    return response
