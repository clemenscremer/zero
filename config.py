import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Define global file paths
# Define the base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")



# Define the simulation data directory relative to the application
SIM_DATA_DIR = os.path.join(BASE_DIR, "sim_data")

# Define the subdirectories within the simulation data directory
TEMP_DIR = os.path.join(SIM_DATA_DIR, "temp")
DOMAIN_DIR = os.path.join(SIM_DATA_DIR, "domain")
INITIAL_DIR = os.path.join(SIM_DATA_DIR, "initial")
SETUP_DIR = os.path.join(SIM_DATA_DIR, "setup")
BOUNDARIES_DIR = os.path.join(SIM_DATA_DIR, "boundaries")
RESULTS_DIR = os.path.join(SIM_DATA_DIR, "results")
FIGURE_DIR = os.path.join(SIM_DATA_DIR, "figures")
EXCEL_FILE_PATH = os.path.join(SIM_DATA_DIR, "simulations.xlsx")


# Create directories if they don't exist
os.makedirs(DOMAIN_DIR, exist_ok=True)
os.makedirs(INITIAL_DIR, exist_ok=True)
os.makedirs(SETUP_DIR, exist_ok=True)
os.makedirs(BOUNDARIES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# GPT 4-o deployment
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
