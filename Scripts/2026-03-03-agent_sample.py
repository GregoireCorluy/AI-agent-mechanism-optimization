import sys
from openai import OpenAI

# --- Client Setup ---
# python -m mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit --port 8080
receptionist_client = OpenAI(base_url="http://localhost:8080/v1", api_key="mlx") 

# --- 1. The Receptionist (Frontend) ---

SYSTEM_PROMPT_FRONTEND = """
You are an expert **Combustion Simulation Consultant**.
Your goal is to actively guide the user to define a "Reduction Configuration".

[REQUIRED PARAMETERS]
1. **Mechanism**: Base file (default assumption: gri30.yaml if not specified).
2. **Fuel(s)**: Target fuel species (e.g., CH4, H2). Handling blends is allowed.
3. **Conditions**: Pressure, Temperature, Equivalence Ratio ranges.
4. **Target Species**: Critical species to predict accurately (e.g., NOx, CO).

[BEHAVIOR GUIDELINES]
- **Active Listening**: Parse the user's input immediately. If the user says "I want to simulate a methane engine," understand that Fuel=CH4 and suggest engine-relevant conditions (High P, High T) automatically.
- **Do NOT Repeat**: If the user provided a parameter, ACKNOWLEDGE it and move to the missing ones. Do not ask for what you already know.
- **Proactive Suggestions**: If the user is vague (e.g., "I don't know the temp range"), propose standard values based on their application (e.g., "For gas turbines, typically 10-30 atm. Shall we start there?").
- **Consulting**: If you notice a missing critical factor (like NOx for emissions), ask: "Are you interested in emission analysis (NOx)?"
- **Final Confirmation**: Only when ALL parameters are reasonably defined or inferred, summarize them clearly and ask "Is this configuration correct?".

[INTERACTION FLOW]
1. Analyze user input.
2. Check which [REQUIRED PARAMETERS] are filled.
3. If parameters are missing:
   - Formulate a question or suggestion for the MISSING parameters only.
   - Group related questions naturally (e.g., ask about P and T together).
4. If all parameters are filled:
   - Output a summary.
   - Ask for final confirmation (YES/NO).
5. If user confirms "YES":
   - Output exactly: [READY]

Start the conversation naturally. Do not be robotic.
"""

# main()

opening_message = """\nHello! I'm your combustion mechanism consultant.\n\
Please tell me about the simulation or target fuel you are working on."""

print(f"\nAgent: {opening_message}")

history = [
    {"role": "system", "content": SYSTEM_PROMPT_FRONTEND},
    {"role": "assistant", "content": opening_message}
]

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]: sys.exit()
        
        history.append({"role": "user", "content": user_input})

        # Chat Interaction
        response = receptionist_client.chat.completions.create(
            model="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            messages=history,
            temperature=0.3, 
            stop=["<|eot_id|>", "<|start_header_id|>user"] 
        )
        
        bot_reply = response.choices[0].message.content
        
        # check [READY] tag
        if "[READY]" in bot_reply:
            clean_reply = bot_reply.replace("[READY]", "").strip()
            if clean_reply:
                print(f"\nAgent: {clean_reply}")
                history.append({"role": "assistant", "content": clean_reply})
            
            print("\n" + "="*60)
            print(">> [SYSTEM] Confirmation received. Handing over to Specialist...")
            print("="*60)
        
        print(f"\nAgent: {bot_reply}")
        history.append({"role": "assistant", "content": bot_reply})
        
    except KeyboardInterrupt:
        sys.exit()