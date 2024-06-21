import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

########## DIRECTORY CONFIG ##########

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPLOAD DIRS 
UPLOAD_SIM_DATA_DIR = os.path.join(BASE_DIR, "uploaded_sim_data")

UPLOAD_DOMAIN_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "domain")
UPLOAD_INITIAL_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "initial")
UPLOAD_SETUP_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "setup")
UPLOAD_BOUNDARIES_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "boundaries")
UPLOAD_RESULTS_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "results")
UPLOAD_FIGURE_DIR = os.path.join(UPLOAD_SIM_DATA_DIR, "figures")
UPLOAD_EXCEL_FILE_PATH = os.path.join(UPLOAD_SIM_DATA_DIR, "simulations.xlsx")

os.makedirs(UPLOAD_SIM_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DOMAIN_DIR, exist_ok=True)
os.makedirs(UPLOAD_DOMAIN_DIR, exist_ok=True)
os.makedirs(UPLOAD_INITIAL_DIR, exist_ok=True)
os.makedirs(UPLOAD_SETUP_DIR, exist_ok=True)
os.makedirs(UPLOAD_BOUNDARIES_DIR, exist_ok=True)
os.makedirs(UPLOAD_RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FIGURE_DIR, exist_ok=True)

# ACTIVE DIRS
ACTIVE_SIM_DATA_DIR = os.path.join(BASE_DIR, "active_sim_data")

ACTIVE_DOMAIN_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "domain")
ACTIVE_INITIAL_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "initial")
ACTIVE_SETUP_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "setup")
ACTIVE_BOUNDARIES_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "boundaries")
ACTIVE_RESULTS_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "results")
ACTIVE_FIGURE_DIR = os.path.join(ACTIVE_SIM_DATA_DIR, "figures")
ACTIVE_EXCEL_FILE_PATH = os.path.join(ACTIVE_SIM_DATA_DIR, "simulations.xlsx")

os.makedirs(ACTIVE_SIM_DATA_DIR, exist_ok=True)
os.makedirs(ACTIVE_DOMAIN_DIR, exist_ok=True)
os.makedirs(ACTIVE_INITIAL_DIR, exist_ok=True)
os.makedirs(ACTIVE_SETUP_DIR, exist_ok=True)
os.makedirs(ACTIVE_BOUNDARIES_DIR, exist_ok=True)
os.makedirs(ACTIVE_RESULTS_DIR, exist_ok=True)
os.makedirs(ACTIVE_FIGURE_DIR, exist_ok=True)

# OUTPUT DIRS
OUTPUT_SIM_DATA_DIR = os.path.join(BASE_DIR, "output_sim_data")

OUTPUT_DOMAIN_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "domain")
OUTPUT_INITIAL_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "initial")
OUTPUT_SETUP_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "setup")
OUTPUT_BOUNDARIES_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "boundaries")
OUTPUT_RESULTS_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "results")
OUTPUT_FIGURE_DIR = os.path.join(OUTPUT_SIM_DATA_DIR, "figures")
OUTPUT_EXCEL_FILE_PATH = os.path.join(OUTPUT_SIM_DATA_DIR, "simulations.xlsx")

os.makedirs(OUTPUT_SIM_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DOMAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_INITIAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_SETUP_DIR, exist_ok=True)
os.makedirs(OUTPUT_BOUNDARIES_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

########## LLM MODEL CONFIG ##########

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
