# A modelling assistant

goal of this project is to create an assistant that can help in all phases of numerical modelling procedure, from
creating a domain, to setting the bathymetry in this domain as well as initial conditions. The assistant can also simple boundary conditions. Via agents the assistant can also run simulations and plot / evaluate results afterwards.


## requirements (status)


### frontend
* Developed in **Streamlit** 
* Components and position
    * **Chat interface** to interact with LLM
    * Visualization window
        * Map: Domain / bathymetry / initial conditions / results (switchable via availability check and dropdown?)
        * Timeseries: Boundary conditions: Left and right of domain

### LLM
* powered by:
    * azure openai for now (=starting proprietary and simple)

* later: modular in different "agents" that can be called by a central instance
    * vision capabilities e.g. from gpt4-v or omni for interpretation of results
    * e.g. setup agent that explicitly knows pfs structure and how to access would increase flexibility a lot, could also use pfs.search("charnock")
    * groq would be great for speed and potential finetuning of open models

#### Steps and Agents (Modular design)
The steps follow a classical modelling workflow. However, they offer flexibility.

1. **General agent**: should be powerful, have most reasoning e.g. to evaluate input from other agents, reason about outputs of 6. (analyze agent).
2. **Domain agent**: generate gridded domain, generate bathymetry (which needs info about discretization), initial condition
3. **Boundary agent**: generate boundary conditions 
4. **Setup agent**: knowledge of pfs file structure and state-awareness (what is already set etc.)
5. **Run agent**: 
6. **Analyze agent**: Knows modelskill and mikeio to extract results
7. (optional) ML agent: Can do forecasting
8. **Reporter agent**: reading from log 


### tools to leverage and todos
* Problem: save_to_dfs called WAAAAY too frequently. Maybe i should just keep a pfs dummy, whether pfs object or ascii, in memory (session_state?) and modify there? Writing when running should suffice. 
* DOING: state awareness: use this as template --> keep track in short term memory of file we are working and parameters
* DOING: define workflow and needed tools clearly (draft below, based on temp_0-1-examplegen-runs.ipynb)
    * create mesh and bathymetry (create_bathymetry(nx, ny, dx, dy, reference_depth, slope_degrees=0.0) --> be open for more complex mesh/bathymetry as well where input is an array of nx,ny with heights)
    * create initial conditions (exchangable with boundary conditions)
    * create boundary conditions (see above, make sure either initial or boundary is included)
    * run simulation (check system specs, look at mesh size and choose best nthreads and nsubdomains --> save compute, maybe ask Gedi on this)
    * evaluate (and save images / condensed results?)
        * plot spatial results at different timesteps
        * plot timeseries
* TODO: whilst creating any file, we could write to an excel with different sheets for boundary, domain, initial, setup, figure.
The individual sheets should contain a version number (hash) for each file. Setups, as they can utilize boundaries, domain, initial should refer to all of those applicable. Figures should refer to setups and contain timesteps, if applicable.
* TODO: let repo bot write the tooling based on workflow and examples in the repo (can just be copied from anthropic page etc, also collect some pfs examples from simple setups and mikeio documentation on reading, altering, writing pfs)
* TODO: research best way to state-awareness / knowledge sharing / logging capabilities (for each agent its own later?)
* TODO: MIKE Zero Linux installation (TODO)
* TODO: find transfer possibility between other model function calls (e.g. anthropic) and groq
* TODO: find great naming conventions for setup-files, boundary conditions, initial conditions, mesh and bathymetry. Think about hash and lookup table for setup files as they otherwise might need to contain too much information in the filename.
* TODO: access to folders (to check if files needed for simulation are created as part of state awareness)
* TODO: vision capabilities
* TODO: pfs awareness
* TODO: (optional) "clean up mode" in the end. Remove unused files from data folder
* (optional) online data access
* (optional) ML timeseries forecast (just based on one dataset e.g. x,y coordinate extracted timeseries)
* (optional) Multiple function calls at the same time
* (optional) extend mesh and bathymetry (e.g. from image) generation to more complex (ask MOM how to generate)


## Suggested initial repo structure (possible subject to iteration later)

Note: a MIKE Zero installation on the system is crucial, however, not directly part of the repo. MIKE Zero files reside in the root folder under MIKE, environmental variables to the FemEngineHD, its GPU counterpart need to be set on the system.

mk-assistant/
├── README.md               <- this file, overview of project status and description
├── requirements.txt        <- could also be a yaml including dependencies
├── src/
│   ├── app.py              <- main streamlit app
│   ├── tools/              <- this is where modules for function calls and descriptions go
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   └── data_utils.py
├── data/                   <- everything for MIKE Zero simulations
│   ├── domain/ 
│   ├── boundaries/
│   ├── initial/
│   └── setups/
│   └── results/
└── logs/
    ├── simulation.log
    └── analysis.log


## Demo
* assistance guiding an unexperienced user through the setup
* assistance e.g. in checking consistency of values for an area
* workflow steps from creating domain to creation of bathymetry, boundary conditions, executing simulation to analysis and reporting. 
* do a second simulation, compare results
* report (to word?)
* Add-on (optional): read initial or bathy from image
* pros: 
    * reproducibility (folder structure, logs, tools)
    * cost efficient (human and compute time, e.g. via efficient nthreads saved)



## Notes

Prompt for Frontend design:

"""
    I want to develop an llm app in streamlit. 
    Some context: 
    The app is supposed to be a companion for numerical modeling and will facilitate setup of a numerical 2d model for shallow water equations. Within the chat interface, the user will be able to define a) a domain (rectangular, gridded), b) bathymetry in this domain c) initial conditions In this domain, 3. boundary conditions for two sides (left and right) of the domain. Results will be displayed on-demand in the chat interface.
    I will implement all this functionality via python functions that the llm can call. 

    ## Task: designing the frontend:
    The app should have following components: 1. a chat interface for interaction with the llm, 2. a graphical overview of a map (showing either the domain, bathymetry, initial conditions) and none or up to two timeseries (boundary conditions) 

    please answer the question in a three step process. 
    Step 1: read the task carefully, isolate critical information for the solution 
    Step 2: formulate 3 solutions. Include reasoning 
    Step 3: evaluate the solutions from step 2, critically reason on each and compare them, and then select the best one in terms of ux and implementation. 
"""

