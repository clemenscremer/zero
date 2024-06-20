import os
import pandas as pd
import mikeio

# Define the base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the simulation data directory relative to the application
SIM_DATA_DIR = os.path.join(BASE_DIR, "sim_data")

SETUP_DIR = os.path.join(SIM_DATA_DIR, "setup")


def get_setup_files():
    """ List all available model setup files in the specified subfolders of the 'sim_data' directory.
    Args:
        None
    Returns:
        dict: A dictionary containing the lists of files for each specified folder.
    """
    file_lists = {}

    # List files in the 'setup' subfolder
    try:
        setup_files = [file for file in os.listdir(SETUP_DIR) if file.endswith('.m21fm')]
        file_lists['setup'] = setup_files
    except FileNotFoundError:
        file_lists['setup'] = []
   
    return file_lists



def get_simfile_parameters(file_path):
    """
    Retrieves modifiable parameters from a PFS numerical setup file.

    Args:
        file_path (str): Path to the PFS file, relative to the 'setup' directory.

    Returns:
        dict: A dictionary containing the requested parameters.
    """
    from mikeio import read_pfs
    pfs = read_pfs(os.path.join(SETUP_DIR, file_path))
    parameters = {}

    start_time_obj = pfs["FemEngineHD"]["TIME"]["start_time"]
    parameters["start_time"] = start_time_obj.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string
    parameters["time_step_interval"] = pfs["FemEngineHD"]["TIME"]["time_step_interval"]
    parameters["number_of_time_steps"] = pfs["FemEngineHD"]["TIME"]["number_of_time_steps"]
    parameters["domain_file"] = pfs["FemEngineHD"]["DOMAIN"]["file_name"]
    parameters["initial_conditions_file"] = pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["INITIAL_CONDITIONS"]["file_name_2D"]
    parameters["initial_surface_elevation"] = pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["INITIAL_CONDITIONS"]["surface_elevation_constant"]
    parameters["boundary_conditions_file"] = pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["BOUNDARY_CONDITIONS"]["CODE_2"]["file_name"]
    parameters["manning_number"] = pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["BED_RESISTANCE"]["MANNING_NUMBER"]["constant_value"]

    return parameters


def save_setup(file_path, parameters):
    """
    Saves a new PFS file with the specified parameters.

    Args:
        file_path (str): Path to template PFS file, relative to the 'setup' directory.
        parameters (dict): Dictionary of parameters to modify, where the keys are the parameter names
                                  and the values are the new values.

    Returns:
        str: Path to the modified PFS file.
    """
    from mikeio import read_pfs

    pfs = read_pfs(os.path.join(SETUP_DIR, file_path))

    # Modify the specified parameters
    for parameter, value in modified_parameters.items():
        if parameter == "start_time":
            pfs["FemEngineHD"]["TIME"]["start_time"] = value
        elif parameter == "time_step_interval":
            pfs["FemEngineHD"]["TIME"]["time_step_interval"] = value
        elif parameter == "number_of_time_steps":
            pfs["FemEngineHD"]["TIME"]["number_of_time_steps"] = value
        elif parameter == "domain_file":
            pfs["FemEngineHD"]["DOMAIN"]["file_name"] = value
        elif parameter == "initial_conditions_file":
            pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["INITIAL_CONDITIONS"]["file_name_2D"] = value
        elif parameter == "initial_surface_elevation":
            pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["INITIAL_CONDITIONS"]["surface_elevation_constant"] = value
        elif parameter == "boundary_conditions_file":
            pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["BOUNDARY_CONDITIONS"]["CODE_2"]["file_name"] = value
        elif parameter == "manning_number":
            pfs["FemEngineHD"]["HYDRODYNAMIC_MODULE"]["BED_RESISTANCE"]["MANNING_NUMBER"]["constant_value"] = value
        else:
            raise ValueError(f"Parameter '{parameter}' is not a valid parameter in the PFS file.")


    # Get the list of existing setup files
    file_lists = list_model_files(return_folders=['setup'])
    setup_files = file_lists['setup']

    # Generate the modified file name with a numbered suffix
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    existing_numbers = [int(f.replace(base_name + "_", "").replace(".m21fm", "")) for f in setup_files if f.startswith(base_name + "_")]
    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    modified_file_path = os.path.join(SETUP_DIR, f"{base_name}_{next_number:03d}.m21fm")

    # Write the modified PFS file
    pfs.write(modified_file_path)

    return modified_file_path



function_descriptions = [
    {
        "name": "get_setup_files",
        "description": "List all available model setup files in the 'setup' subfolder of the 'sim_data' directory.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "returns": {
            "type": "object",
            "description": "A dictionary containing the lists of setup files.",
            "properties": {
                "setup": {
                    "type": "array",
                    "description": "List of *.m21fm files in the 'setup' subfolder."
                }
            }
        }
    },
    {
        "name": "get_simfile_parameters",
        "description": "Retrieves modifiable parameters from a PFS numerical setup file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PFS file, relative to the 'setup' directory."
                }
            },
            "required": [
                "file_path"
            ]
        },
        "returns": {
            "type": "object",
            "description": "A dictionary containing the requested parameters.",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "The start time of the simulation."
                },
                "time_step_interval": {
                    "type": "number",
                    "description": "The time step interval in seconds."
                },
                "number_of_time_steps": {
                    "type": "integer",
                    "description": "The number of time steps in the simulation."
                },
                "domain_file": {
                    "type": "string",
                    "description": "The file name of the domain file (mesh and bathymetry)."
                },
                "initial_conditions_file": {
                    "type": "string",
                    "description": "The file name of the initial conditions file."
                },
                "initial_surface_elevation": {
                    "type": "number",
                    "description": "The constant surface elevation in meters."
                },
                "boundary_conditions_file": {
                    "type": "string",
                    "description": "The file name of the boundary conditions file."
                },
                "manning_number": {
                    "type": "number",
                    "description": "The constant Manning number."
                }
            }
        }
    },
    {
        "name": "save_setup",
        "description": "Saves a new PFS file with the specified parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to template PFS file, relative to the 'setup' directory."
                },
                "parameters": {
                    "type": "object",
                    "description": "Dictionary of parameters to modify, where the keys are the parameter names and the values are the new values."
                }
            },
            "required": [
                "file_path",
                "parameters"
            ]
        },
        "returns": {
            "type": "string",
            "description": "Path to the modified PFS file."
        }
    }
]


available_functions = {
    "get_setup_files": get_setup_files,
    "get_simfile_parameters": get_simfile_parameters,
    "save_setup": save_setup
}