import os
import re
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
    problem_statement: str | None = Field(
        None, 
        description="The supply chain or routing problem statement to be analyzed by the agent.",
        examples=["Optimize warehouse inventory levels given a 20% spike in regional shipping delays."]
    )
    issue: str | None = Field(
        None,
        description="Fallback field for frontend payload integration."
    )
    priority: str | None = Field(
        None,
        description="Optional priority indicator sent by the frontend."
    )
    autonomy: str | None = Field(
        None,
        description="Optional autonomy level slider target."
    )

# Define response schema
class AgentResponse(BaseModel):
    response: str = Field(
        ...,
        description="The full raw text response from the Gemini model."
    )
    analysis: str | None = Field(
        None,
        description="Extracted content of the [ANALYSIS] section."
    )
    strategy: str | None = Field(
        None,
        description="Extracted content of the [STRATEGY] section."
    )
    action_plan: list[str] | None = Field(
        None,
        description="List of action items extracted from the [ACTION PLAN] section."
    )
    actionPlan: list[str] | None = Field(
        None,
        description="CamelCase copy of action_plan for frontend compatibility."
    )

def parse_sections(text: str):
    """
    Utility function to parse the generated response into separate sections:
    [ANALYSIS], [STRATEGY], and [ACTION PLAN].
    """
    # Case-insensitive regex matches to handle possible formatting variations from the LLM
    analysis_match = re.search(r'\[ANALYSIS\](.*?)(\[STRATEGY\]|\[ACTION PLAN\]|$)', text, re.DOTALL | re.IGNORECASE)
    strategy_match = re.search(r'\[STRATEGY\](.*?)(\[ANALYSIS\]|\[ACTION PLAN\]|$)', text, re.DOTALL | re.IGNORECASE)
    action_plan_match = re.search(r'\[ACTION PLAN\](.*)', text, re.DOTALL | re.IGNORECASE)
    
    # Retrieve matches or empty string
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    strategy = strategy_match.group(1).strip() if strategy_match else ""
    action_plan_text = action_plan_match.group(1).strip() if action_plan_match else ""
    
    # Clean up and split the Action Plan lines into discrete list items
    action_plan = []
    if action_plan_text:
        # Split by newlines and discard empty lines
        lines = [line.strip() for line in action_plan_text.split('\n') if line.strip()]
        for line in lines:
            # Strip standard markdown bullet formats (-, *, 1., [ ], etc.)
            cleaned = re.sub(r'^(\-\s*|\*\s*|\d+\.\s*|\[\s*\]\s*)', '', line).strip()
            if cleaned:
                action_plan.append(cleaned)
                
    # Fallback if the parser fails to match sections
    if not analysis and not strategy and not action_plan:
        analysis = text
        strategy = "Refer to the main analysis."
        action_plan = ["Review full architect recommendations."]
        
    return analysis, strategy, action_plan

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
    
    Accepts problem_statement (or issue), reads GEMINI_API_KEY from environment variables, 
    sends the problem statement to gemini-2.5-flash with strict system instructions, 
    parses out structured sections, and returns the response.
    """
    # Support both problem_statement and issue to be compatible with frontend configurations
    problem = request.problem_statement or request.issue
    if not problem:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either 'problem_statement' or 'issue' must be provided in the request body."
        )

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
            contents=problem,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2, # Low temperature for more analytical and deterministic architecture plans
            )
        )
        
        response_text = response.text
        if not response_text:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Received an empty response from the Gemini API."
            )
            
        # Parse the structured sections from the text response
        analysis, strategy, action_plan = parse_sections(response_text)
            
        return AgentResponse(
            response=response_text,
            analysis=analysis,
            strategy=strategy,
            action_plan=action_plan,
            actionPlan=action_plan
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API request failed: {str(e)}"
        )
