import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import AzureOpenAI
import json
import altair as alt
from func_helpers import available_functions, function_descriptions 
import matplotlib
import pandas as pd
import os
import datetime

# import paths from central config
from config import SIM_DATA_DIR, TEMP_DIR, EXCEL_FILE_PATH
# import api client from central config
from config import client, DEPLOYMENT_NAME

from PIL import Image
# Load custom avatar
assistant_avatar = Image.open("simple_mk-assistant/img/dalle_bot.webp")
usr_avatar = Image.open("simple_mk-assistant/img/dalle_engineer.webp")

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

def save_message_history():
    """Save the current message history to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"message_history_{timestamp}.json"
    filepath = os.path.join(TEMP_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(st.session_state.messages, f, indent=2)

    return filepath

def load_message_history(filepath):
    """Load a message history from a JSON file."""
    with open(filepath, 'r') as f:
        loaded_messages = json.load(f)

    st.session_state.messages = loaded_messages

def get_message_history_files():
    """Get a list of message history files from TEMP_DIR, sorted by recency."""
    files = [f for f in os.listdir(TEMP_DIR) if f.startswith("message_history_") and f.endswith(".json")]
    full_paths = [os.path.join(TEMP_DIR, f) for f in files]
    sorted_files = sorted(full_paths, key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in sorted_files]

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
# Load and display the header image
st.image("simple_mk-assistant/img/zero.png")

with st.expander("ℹ️ About"):
    st.write(
        """
        This is a conversational AI assistant that helps to build, run 2D numerical fluid simulations with DHIs MIKE21FM software and evaluate them. 
        For this you have access to a set of functions that let you get an overview of available files, create and modify numerical setups, create mesh and bathymetry, 
        create initial and boundary conditions, run and evaluate simulations.

        It is also capable to guide the user through each step of the process and answer questions about parameters, available simulations etc.

        Implemented functionality
        * simulation file awareness `list_model_files`. E.g. *"please provide an overview of available setup files"*
        * read parameters from pfs file `get_pfs_parameters` (currently limited to few). E.g. *"what are the parameters of the setup file?"*
        * modify parameters `modify_parameters`. E.g. *"please change the roughness to 12"*
        * create mesh and bathymetry `create_mesh_bathymetry`. E.g. "let's create a new mesh and bathymetry. I would like a reference depth of -20 m, 100 elements in x and 30 in y-direction with 10 m grid cell size in both directions"
        * create plotting for mesh and bathymetry `plot_mesh_bathymetry` (check for availability of files first). E.g. "can you plot this?"
        * initial conditions `create_surface_elevation`
        * plot initial conditions `plot_mesh_bathy` (reusing the plotting function for mesh and bathymetry). E.g. "can we create an initial condition with the same nx,ny and a wave of 5 m height and 200 m width on the left?"
        * create a setup 
        * run simulation(s) `simulate`. Requires `mikesimulation.py`. E.g. "can you run sim_.m21fm?"
        * create figures from results from notebook `mike_workflow.ipynb`, `func_helpers.plot_results(simulation, n_times=3)`
        * evaluation of figures with analyze_images(image_files, added_context)`
        * saving message history and loading them from disk
        * reporting in word `generate_report`
        * ADD: extract timeseries data from simulation results
        * ADD: ML prediction from timeseries data
        """
    )

# -------------------------------------------
# message handling and system prompt
# -------------------------------------------
def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)  

sys_prompt = """You are an assistant that helps to build, run 2D numerical fluid simulations with DHIs MIKE21FM software and evaluate them. 
For this you have access to a set of functions that let you
1. get an overview of available files, whether setup or boundary conditions or else (list_model_files)
2. open setup files and return parameters (get_pfs_parameters), 
3. create mesh and bathymetry (create_mesh_bathymetry), 
4. create initial conditions (create_surface_elevation),
5. modify parameters 
6. write to new setup files (save_setup), which you ONLY use when explicitly prompted by user
7. run simulations (simulate).
8. create figures from simulation results (plot_results)
9. evaluate result figures and compare multiple results (analyze_images)
10. write a report and export to word (generate_report)

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

# Create a container for messages
message_container = st.container()

# Display the prior chat messages
with message_container:
    for message in st.session_state.messages:  
        if message["role"] == "user":
            with st.chat_message("user", avatar=usr_avatar):
                st.markdown(message["content"])        
        elif message["role"] == "assistant" and message["content"] != None:
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(message["content"])
        # else, if message role is function or content is none then pass
        else:
            pass

# handle user input
if prompt := st.chat_input("Your question"):
    add_to_message_history("user", prompt)
    with message_container:
        with st.chat_message("user", avatar=usr_avatar):
            st.markdown(prompt)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with message_container:
        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=st.session_state.messages,
                    temperature=0.0,
                    stream=False,
                    functions=function_descriptions,
                    function_call="auto", 
                    )
                response_message = response.choices[0].message

                if response_message.function_call:    
                    function_name = response_message.function_call.name
                    function_to_call = available_functions[function_name] 
                    function_args = json.loads(response_message.function_call.arguments)
                    with st.status(f"Calling function {function_name} with arguments below:"):
                        st.write(function_args)
                        function_response = function_to_call(**function_args)

                    st.session_state.messages.append(
                        {
                            "role": response_message.role,
                            "function_call": {
                                "name": function_name,
                                "arguments": response_message.function_call.arguments,
                            },
                            "content": None
                        }
                    )

                    if isinstance(function_response, matplotlib.figure.Figure):
                        st.pyplot(function_response)
                        add_to_message_history(response_message.role, "Plotting the result")
                    elif function_name == "get_pfs_parameters":
                        add_to_message_history(response_message.role, function_response)
                        st.session_state.params = function_response
                    elif function_name in ["plot_mesh_bathy", "plot_results"]:
                        st.pyplot(function_response)
                        add_to_message_history(response_message.role, "Here is the plot you requested.")
                    elif function_name == "modify_parameters":
                        add_to_message_history(response_message.role, function_response)
                        if isinstance(function_response, dict):
                            for key, value in function_response.items():
                                st.session_state.params[key] = value
                        else:
                            st.error("Unexpected response format from modify_parameters function")
                    elif function_name == "generate_report":
                        report_result = function_response
                        st.success(report_result["message"])
                        with open(report_result["file_path"], "rb") as file:
                            st.download_button(
                                label="Download Report",
                                data=file.read(),
                                file_name=os.path.basename(report_result["file_path"]),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        add_to_message_history("assistant", "I've generated the report based on our conversation. You can download it using the button above.")
                    elif function_name == "analyze_images":
                        response_message = function_response["response_message"]
                        figure = function_response["figure"]
                        add_to_message_history(response_message.role, response_message.content)
                        st.pyplot(figure)
                    else:
                        add_to_message_history(response_message.role, function_response)

                    second_response = client.chat.completions.create(
                            model=DEPLOYMENT_NAME,
                            messages=st.session_state.messages,
                            temperature=0.0,
                        )
                    second_response_message = second_response.choices[0].message

                    add_to_message_history(second_response_message.role, second_response_message.content)
                    st.markdown(second_response_message.content)
                else:
                    st.markdown(str(response_message.content))
                    add_to_message_history(response_message.role, response_message.content) 

with st.sidebar:
    if st.button("Save chat history"):
        saved_file = save_message_history()
        st.success(f"Chat history saved to {saved_file}")

    st.write("Load chat history:")
    history_files = get_message_history_files()
    if history_files:
        selected_file = st.selectbox("Select a file to load", history_files)
        if st.button("Load selected history"):
            load_message_history(os.path.join(TEMP_DIR, selected_file))
            st.success(f"Chat history loaded from {selected_file}")
            st.experimental_rerun()
    else:
        st.write("No saved chat histories found.")

    if st.button("Clear chat history"):
        st.session_state.messages = system_message
        st.session_state.messages.append(
            {"role": "assistant", "content": "How can I assist?"} # initial message
        )
        st.experimental_rerun()

    with st.expander("Full message history"):
        st.write(len(st.session_state.messages))
        st.write(st.session_state.messages)