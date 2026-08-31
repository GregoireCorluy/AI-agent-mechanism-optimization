import json
from CombustionAgent.agent import Agent
from CombustionAgent.parameters import InputParameters
from CombustionAgent.prompts import get_chat_prompt, get_fill_prompt, get_retrieve_prompt, get_router_prompt, get_update_prompt, get_verify_prompt

# TO DO:
# - Fourth model for suggestions
# - Add temperature for models
# - chat with different iterations of the json, can receive partially filled in json as input
# - Ask at the end if the suggestion of parameters is correct
# - Improve description of Pydantic and check what the Pydantic verification is doing
# - Explore other LLMs
# - Constrain output generation for information retrieval
# - Have an LLM that iteratively specifically converts values
# - Check how to manage the history and different messages
# - let the agent think first: first generation step is analyzing the input and second generation provides the structured output; use another LLM for performing the conversion; use Python to perform the deterministic conversion and LLM just identifies the fields with the units
# - Verification also with a structured json
# - Python code to do the conversion from one unit to another, or use LLM
# - Verify again what the update has done?
# - Add a memory: notes to keep for later, keep messages of the user (?)
# - LLM router
# - convert units etc. at the end with a deterministic method
# - Ask at the end if the user agrees with the retrieved parameters
# - Create Langgraph which will choose the possibilities in function of the current state, state-dependent action
# - One LLM to take notes/memory
# - Explain to the user what is happening behind the scenes
# - Message about how the system works
# - Keep conversation history
# - ask at the end if the configuration is correct
# - have a longer discussion with multiple turns to understand what the user wants and then do the retrieval
# - to speed up the loop, no update if already indicated that there is no modification necessary
# - router, let it think first to make its decision?
# - to consider: if the message of the user is both about the simulation parameters and a question, should be handled afterwards by the chat model?
# - Fourth model for suggestions?
# - maybe a start and end value for the temperature and pressure

# TO DO 28th of August 2026
# X Function to handle REAL END
# X change name to process_history
# X remove history for all LLMs except the conversation LLM
# X update the state keeping only the input parameters, but setting the process history back to null
# X let's make a scan with ChatGPT if he finds main errors
# - reorganize code in different scripts when langgraph works
# - check to do's above

# TO DO later
# - Explain the choise of the different parameters
# - Update parameters after verification retrieval only if something to be modified

# POSSIBLE PROBLEMS:
# - number of new max tokens too small and limits the generated output of the LLM

# QUESTIONS:
# - Python or LLM to convert values?

# NOTES:
# - LLM working better when it can "reason" and be wordy


model_id = "Models/Llama-3.1-8B-Instruct"

schema = InputParameters.model_json_schema()

opening_message = "\nHello, I'm your combustion mechanism consultant.\nI will try to provide you the best chemical mechanism for your application.\nCan you describe the simulation you would like to perform?"

chatting_agent = Agent(model_id, [get_chat_prompt(), get_retrieve_prompt(schema), get_verify_prompt(), get_update_prompt(schema), get_fill_prompt(schema), get_router_prompt()], opening_message)
chatting_agent.chat()