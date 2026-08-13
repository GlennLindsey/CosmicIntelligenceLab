import json
import ollama

from tools.simbad_lookup import simbad_lookup
from tools.mast_search import mast_search

# ---------------------------------------------------------
# Tools available to Cosmic AI
# ---------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "simbad_lookup",
            "description": (
                "Query the SIMBAD astronomical database for factual "
                "information about a known astronomical object."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Astronomical object name or identifier, "
                            "such as M51, M31, NGC 5195, or NGC 1275."
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mast_search",
            "description": (
                "Search the MAST astronomical archive for observations "
                "of an astronomical object. Reports the missions found "
                "and the number of JWST and HST observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Astronomical object name or identifier, "
                            "such as M51, M31, or NGC 5195."
                        ),
                    }
                },
                "required": ["name"],
            },
        },
    },
]


# ---------------------------------------------------------
# Conversation
# ---------------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are Cosmic AI, a local astronomical research assistant. "
            "You have access to astronomical databases through tools. "
            "IMPORTANT EVIDENCE RULES: "
            "You may only describe something as a RETRIEVED FACT "
            "if it appears explicitly in an Evidence Record returned "
            "by one of the tools. "
            "Do not infer or invent identifiers, historical facts, "
            "classifications, distances, names, relationships, dates, "
            "or other factual claims unless they are explicitly "
            "supported by the available Evidence Records. "
            "If information is not present in the Evidence Records, "
            "say that it is NOT ESTABLISHED BY THE AVAILABLE EVIDENCE. "
            "Keep three categories separate: "
            "(1) EVIDENCE — facts explicitly returned by astronomical "
            "databases; "
            "(2) INTERPRETATION — conclusions reasonably derived "
            "from the evidence; "
            "(3) NOT ESTABLISHED — information for which the available "
            "evidence provides no support. "
            "Never present an interpretation as if it were a retrieved "
            "database fact. "
            "When producing a final answer, clearly identify the "
            "source of important evidence."
        ),
    },
    {
        "role": "user",
        "content": (
            "Investigate M51. First identify the object using SIMBAD. "
            "Then search MAST for archival observations, paying "
            "particular attention to JWST and HST observations. "
            "Finally summarize what you found."
        ),
    },
]

# ---------------------------------------------------------
# Agent loop
# ---------------------------------------------------------

step = 1

while True:

    print(f"\n{'=' * 60}")
    print(f"AGENT STEP {step}")
    print(f"{'=' * 60}")

    response = ollama.chat(
        model="nemotron-3-nano:4b",
        messages=messages,
        tools=tools,
    )

    print("\nMODEL RESPONSE:")
    print(response.message)

    # Add Nemotron's response to the conversation.
    messages.append(response.message)

    # -----------------------------------------------------
    # No tool call = the agent has finished
    # -----------------------------------------------------

    if not response.message.tool_calls:

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(response.message.content)

        break

    # -----------------------------------------------------
    # Process each requested tool call
    # -----------------------------------------------------

    for call in response.message.tool_calls:

        function_name = call.function.name
        arguments = call.function.arguments

        print("\nTOOL CALLED:")
        print(function_name)

        print("\nARGUMENTS:")
        print(arguments)

        # -------------------------------------------------
        # SIMBAD
        # -------------------------------------------------

        if function_name == "simbad_lookup":

            result = simbad_lookup(arguments["name"])

        # -------------------------------------------------
        # MAST
        # -------------------------------------------------

        elif function_name == "mast_search":

            result = mast_search(arguments["name"])

        # -------------------------------------------------
        # Unknown tool
        # -------------------------------------------------

        else:

            result = {"error": f"Unknown tool requested: {function_name}"}

        print("\nTOOL RESULT:")
        print(result)

        # -------------------------------------------------
        # Return the tool result to Nemotron
        # -------------------------------------------------

        messages.append(
            {
                "role": "tool",
                "tool_name": function_name,
                "content": json.dumps(result),
            }
        )

    step += 1
