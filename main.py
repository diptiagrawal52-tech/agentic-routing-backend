import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize FastAPI application
app = FastAPI(
    title="Enterprise Agentic Routing Engine API",
    description="Backend service for Project 3: An Enterprise Agentic Routing Engine using FastAPI and Gemini 2.5 Flash.",
    version="1.0.0"
)

# Configure CORS Middleware
# Enabled to allow secure fetch requests from any Vercel frontend or external web client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Set to False when allowing wildcard (*) origins to prevent CORS errors
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class AgentRequest(BaseModel):
    problem_statement: str = Field(
        ..., 
        description="The supply chain or routing problem statement to be analyzed by the agent.",
        examples=["Optimize warehouse inventory levels given a 20% spike in regional shipping delays."]
    )

# Define response schema
class AgentResponse(BaseModel):
    response: str = Field(
        ...,
        description="The structured output from the Enterprise Supply Chain Architect containing [ANALYSIS], [STRATEGY], and [ACTION PLAN]."
    )

@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    """
    Health check and root info endpoint.
    """
    return {
        "status": "online",
        "service": "Enterprise Agentic Routing Engine API",
        "model": "gemini-2.5-flash"
    }

@app.post("/run-agent", response_model=AgentResponse, status_code=status.HTTP_200_OK)
async def run_agent(request: AgentRequest):
    """
    Executes the agentic workflow as an Enterprise Supply Chain Architect.
    
    Reads GEMINI_API_KEY from environment variables, sends the problem statement 
    to gemini-2.5-flash with strict system instructions, and returns the response.
    """
    # Read Gemini API Key from environment variables
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured in the server environment variables."
        )

    # Initialize the Google GenAI Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize the Gemini client: {str(e)}"
        )

    # Define the system instructions for the Enterprise Supply Chain Architect
    system_instruction = (
        "You are an expert Enterprise Supply Chain Architect.\n\n"
        "Analyze the user's problem statement and provide a detailed, optimal solution. "
        "You must structure your response using exactly three sections: [ANALYSIS], [STRATEGY], and [ACTION PLAN].\n\n"
        "Follow these structural guidelines strictly:\n"
        "1. Start with the [ANALYSIS] section. Detail the problem's scope, bottlenecks, risks, and critical constraints.\n"
        "2. Follow with the [STRATEGY] section. Propose architectural changes, supply chain workflows, routing optimization logic, and risk mitigation strategies.\n"
        "3. Conclude with the [ACTION PLAN] section. Provide a clear, phased, step-by-step roadmap for implementation.\n\n"
        "Strict formatting rule: Output only these three sections. Do not include any greeting, intro, outro, or additional markdown headings above [ANALYSIS]."
    )

    try:
        # Request content generation using gemini-2.5-flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.problem_statement,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2, # Low temperature for more analytical and deterministic architecture plans
            )
        )
        
        if not response.text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Received an empty response from the Gemini API."
            )
            
        return AgentResponse(response=response.text)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API request failed: {str(e)}"
        )
