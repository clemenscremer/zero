import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import shutil
import os

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

# remove all dirs before creating
print("SYS INFO: Removing all files in active dir")
shutil.rmtree(ACTIVE_SIM_DATA_DIR, ignore_errors=True)

os.makedirs(ACTIVE_SIM_DATA_DIR, exist_ok=True)
os.makedirs(ACTIVE_DOMAIN_DIR, exist_ok=True)
os.makedirs(ACTIVE_INITIAL_DIR, exist_ok=True)
os.makedirs(ACTIVE_SETUP_DIR, exist_ok=True)
os.makedirs(ACTIVE_BOUNDARIES_DIR, exist_ok=True)
os.makedirs(ACTIVE_RESULTS_DIR, exist_ok=True)
os.makedirs(ACTIVE_FIGURE_DIR, exist_ok=True)

# TEMP DIRS
TEMP_SIM_DATA_DIR = os.path.join(BASE_DIR, "temp_sim_data")

TEMP_DOMAIN_DIR = os.path.join(TEMP_SIM_DATA_DIR, "domain")
TEMP_INITIAL_DIR = os.path.join(TEMP_SIM_DATA_DIR, "initial")
TEMP_SETUP_DIR = os.path.join(TEMP_SIM_DATA_DIR, "setup")
TEMP_BOUNDARIES_DIR = os.path.join(TEMP_SIM_DATA_DIR, "boundaries")
TEMP_RESULTS_DIR = os.path.join(TEMP_SIM_DATA_DIR, "results")
TEMP_FIGURE_DIR = os.path.join(TEMP_SIM_DATA_DIR, "figures")
TEMP_EXCEL_FILE_PATH = os.path.join(TEMP_SIM_DATA_DIR, "simulations.xlsx")

# remove all dirs before creating
print("SYS INFO: Removing all files in temp dir")
shutil.rmtree(TEMP_SIM_DATA_DIR, ignore_errors=True)

os.makedirs(TEMP_SIM_DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DOMAIN_DIR, exist_ok=True)
os.makedirs(TEMP_INITIAL_DIR, exist_ok=True)
os.makedirs(TEMP_SETUP_DIR, exist_ok=True)
os.makedirs(TEMP_BOUNDARIES_DIR, exist_ok=True)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_FIGURE_DIR, exist_ok=True)

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

########## COPYING FILES ##########

# for now I will copy all the data in the base dir into the upload_sim dir upon streamlit start
# in future this will be done be part of the upload workflow

print("SYS INFO: Copying files from upload dir to active dir")
shutil.copytree(UPLOAD_SIM_DATA_DIR, ACTIVE_SIM_DATA_DIR, dirs_exist_ok=True)

print("SYS INFO: Copying files from active dir to temp dir")
shutil.copytree(ACTIVE_SIM_DATA_DIR, TEMP_SIM_DATA_DIR, dirs_exist_ok=True)


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
