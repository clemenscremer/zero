import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI
import json
import altair as alt
from func_helpers import available_functions, function_descriptions # I am "outsourcing" the function implementations and descriptions to a separate file
#from pfs_helpers import available_functions, function_descriptions
import matplotlib
import pandas as pd
import os
#
# Define the simulation data directory
SIM_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data")
EXCEL_FILE_PATH = os.path.join(SIM_DATA_DIR, "simulations.xlsx")
#

# -------------------------------------------
# import environment variables from .env if not in st.session_state
# -------------------------------------------
if "azure_endpoint" not in st.session_state.keys():
    load_dotenv(find_dotenv(".env"))
    import os

    azure_endpoint = os.getenv("GPT35_AZURE_ENDPOINT")
    api_key = os.getenv("GPT35_AZURE_API_KEY")
    api_version = "2023-10-01-preview"#"2023-09-01-preview"#"2023-05-15"
    
    # create client executing llm
    client = AzureOpenAI(
        azure_endpoint = azure_endpoint,
        api_key = api_key,
        api_version = api_version
    )
    model = "gpt35turbo_autoupdate" # = deployment name
    #model = "gpt-4-turbo"  


# -------------------------------------------
# frontend helper functions
# -------------------------------------------
def refresh_simulation_overview():
    """
    Refresh the simulation overview table in the Streamlit app.
    """
    try:
        sim_xl = pd.read_excel(EXCEL_FILE_PATH)
        st.session_state.simulation_overview = sim_xl
    except FileNotFoundError:
        st.session_state.simulation_overview = pd.DataFrame()
    
    
# -------------------------------------------
# streamlit page 
# -------------------------------------------
st.set_page_config(
    page_title="Chat",
    page_icon="〰️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)
with st.expander("ℹ️ MKAssistant"):
    st.write(
        """
        This is a conversational AI assistant that helps to build, run 2D numerical fluid simulations with DHIs MIKE21FM software and evaluate them. 
        For this you have access to a set of functions that let you get an overview of available files, create and modify numerical setups, create mesh and bathymetry, 
        create initial and boundary conditions, run and evaluate simulations.

        It is also capable to guide the user through each step of the process and answer questions about parameters, available simulations etc.

        Implemented functionality
        * simulation file awareness `list_model_files`. E.g. *"please provide an overview of available setup files"*
        * read parameters from pfs file `get_pfs_parameters` (currently limited to few). E.g. *"what are the parameters of the setup file?"*
        * modify parameters, write to pfs file and keep track of available simulations in excel `modify_pfs_parameters`. 
        E.g. *"please change the roughness to 12"*
        * create mesh and bathymetry `create_mesh_bathymetry`. E.g. "let's create a new mesh and bathymetry. I would like a reference depth of -20 m, 100 elements in x and 30 in y-direction with 10 m grid cell size in both directions"
        * create plotting for mesh and bathymetry `plot_mesh_bathymetry` (check for availability of files first). E.g. "can you plot this?"
        * initial conditions `create_surface_elevation`
        * plot initial conditions `plot_mesh_bathy` (reusing the plotting function for mesh and bathymetry). E.g. "can we create an initial condition with the same nx,ny and a wave of 5 m height and 200 m width on the left?"
        * UPGRADE: central config file providing API access and paths
        * ADD: run simulation(s) from notebook `mike_workflow.ipynb`, `simulate(simfile_name)`
        * ADD: create figures from results from notebook `mike_workflow.ipynb`, `func_helpers.plot_results(simulation, n_times=3)`
        * ADD: evaluation of figures like in `mike_workflow.ipynb`,`evaluation_tools.analyze_images(image_files, added_context)`
        * TODO: add a "remember_this" function to write to a log file or similar, could also be automatically triggered later
        * TODO: awareness of the setup the user is working on (temp_setup.txt?). User should have the possibility to modify anything and only then save a new setup file to run. 
        * TODO: chat widget? https://github.com/Mintplex-Labs/anything-llm/blob/master/embed/README.md
        """
    )
with st.expander("ℹ️ Simulations overview"):
    if "simulation_overview" in st.session_state:
        st.write(st.session_state.simulation_overview)
    else:
        st.write("No simulation data available")

    if st.button("Refresh Simulation Overview"):
        refresh_simulation_overview()

# -------------------------------------------
# message handling and system prompt
# -------------------------------------------
def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)  


sys_prompt = """You are an assistant that helps to build, run 2D numerical fluid simulations with DHIs MIKE21FM software and evaluate them. 
For this you have access to a set of functions that let you
1. get an overview of available files, whether setup or boundary conditions or else 
2. open setup files and return parameters, 
3. create mesh and bathymetry, 
4. create initial conditions,
5. create boundary conditions,
6. write to new setup files (save_to_pfs), which you ONLY use when explicitly prompted by user
7. run and evaluate simulations.

Anything else, e.g. modification of loaded parameters can be done in normal conversations. 
In case a user needs guidance on building a numerical setup, you can walk him step-by-step through the setup process by either assuming or asking the user for 
parameters you need for the function calls. Always let the user know if you are assuming some parameter and justify your assumption.
""" 
system_message= [
    {"role": "system", 
     "content": sys_prompt
    }]

# Initialize the chat messages history
if "messages" not in st.session_state.keys():  
    st.session_state.messages = system_message
    st.session_state.messages.append(
        {"role": "assistant", "content": "How can i assist?"} # initial message
        )


# Display the prior chat messages if the role is not function #
for message in st.session_state.messages:  
    if message["role"] == "user":
        with st.chat_message("user"):#, avatar=user_avatar):
            st.write(message["content"])        
    elif message["role"] == "assistant" and message["content"] != None:
        with st.chat_message("assistant"):#, avatar=bot_avatar):
            st.write(message["content"])
    # else, if message role is function or content is none then pass
    else:
        pass

# handle user input
if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    add_to_message_history("user", prompt)
    with st.chat_message("user"):#, avatar=user_avatar):
        st.write(prompt)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):#, avatar=bot_avatar):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=0.0,
                stream=False,
                functions=function_descriptions,
                function_call="auto", 
                )
            response_message = response.choices[0].message

            if response_message.function_call:    
                # Call the function. TODO: Include error handling for non-valid JSON response
                function_name = response_message.function_call.name
                function_to_call = available_functions[function_name] 
                # get well formatted function arguments from llm response
                function_args = json.loads(response_message.function_call.arguments)
                # call the function and expose it and arguments in streamlit
                with st.status(f"Calling function {function_name} with arguments below:"):
                    st.write(function_args)
                    function_response = function_to_call(**function_args)

                # Add the assistant response and function response to the messages
                # NOTE: here not using the add_to_message_history function because function message format is different
                st.session_state.messages.append( # adding assistant response to messages
                    {
                        "role": response_message.role,
                        "function_call": {
                            "name": function_name,
                            "arguments": response_message.function_call.arguments,
                        },
                        "content": None
                    }
                )
                # NOTE: manually catching special types of function responses
                st.write(str(type (function_response)))
                # Handle different types of function responses
                if type(function_response) == 'matplotlib.figure.Figure':
                    st.pyplot(function_response)
                    add_to_message_history(response_message.role, "Plotting the result")
                else:
                    st.write(function_response)
                    add_to_message_history(response_message.role, function_response)

                    
                # Call the API again to get the final response from the model
                second_response = client.chat.completions.create(
                        model=model,
                        messages=st.session_state.messages,
                        temperature=0.0,
                        #stream=False,
                    )
                second_response_message = second_response.choices[0].message

                add_to_message_history(second_response_message.role, second_response_message.content)
                st.write(second_response_message.content)
            else:
                st.write(str(response_message.content))
                add_to_message_history(response_message.role, response_message.content) 
else:
    pass



with st.sidebar:
    # add button to clear chat history
    if st.button("Clear chat history"):
        st.session_state.messages = system_message
        st.session_state.messages.append(
            {"role": "assistant", "content": "How can i assist?"} # initial message
            )
    
    st.title("Full message history")
    st.write(len(st.session_state.messages))
    st.write(st.session_state.messages)
    