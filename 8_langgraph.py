from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict , Annotated
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import operator
from langsmith import traceable

load_dotenv()

os.environ['LANGSMITH_PROJECT'] = 'LangGraph Essay Evaluation'


api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key  
)

class EvaluationSchema(BaseModel):

    feedback: str = Field(description="Detailed feedback for the essay")
    score:  int = Field(description="Score out of 10 for the essay", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = """In today's rapidly evolving world, technology plays a pivotal role in shaping our lives. 
From communication to healthcare, technological advancements have revolutionized various sectors, 
enhancing efficiency, accuracy, and accessibility. The rise of artificial intelligence, automation, 
and data-driven systems has not only simplified everyday tasks but also transformed industries such 
as education, transportation, and finance.

However, while these innovations bring undeniable benefits, they also introduce new challenges that 
demand careful consideration. Issues like data privacy, job displacement, and digital addiction 
highlight the importance of responsible innovation. It is essential to strike a balance between 
embracing progress and preserving the human values that define our society. 

Education systems must adapt to equip individuals with the skills needed to thrive in a tech-driven 
economy, while policymakers should ensure that ethical standards guide the development and deployment 
of emerging technologies. As we continue to navigate this digital age, our collective responsibility 
is to use technology as a tool for empowerment, not exploitation, fostering a harmonious coexistence 
between humans and machines."""


prompt = f"""Evaluate the following essay based on the criteria of Clarity of Thought, Language, and Depth of Analysis.
{essay}
"""


class EssayEvaluationState(TypedDict):

    essay: str
    clarity_feedback: str
    language_feedback: str
    depth_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add] # here add is a reducer function
    average_score: float


# ----------- Traced node Functions -----------

@traceable(name="evaluate_clarity")
def evaluate_clarity(state: EssayEvaluationState) -> EssayEvaluationState:
    prompt = f"""Evaluate the following essay for Clarity of Thought:
             {state['essay']} Provide detailed feedback and a score out of 10."""
    result = structured_model.invoke(prompt)
    clr_feedback = result.feedback
    clr_score = result.score
    return {
        'clarity_feedback': clr_feedback,
        'individual_scores': [clr_score]
    }

@traceable(name="evaluate_language")
def evaluate_language(state: EssayEvaluationState) -> EssayEvaluationState:
    prompt = f"""Evaluate the following essay for Language Quality:
             {state['essay']} Provide detailed feedback and a score out of 10."""
    result = structured_model.invoke(prompt)
    lang_feedback =  result.feedback
    lang_score = result.score
    return {
        'language_feedback': lang_feedback,
        'individual_scores': [lang_score]
    }

@traceable(name="evaluate_depth")
def evaluate_depth(state: EssayEvaluationState) -> EssayEvaluationState:
    prompt = f"""Evaluate the following essay for Depth of Analysis:
             {state['essay']} Provide detailed feedback and a score out of 10."""
    result = structured_model.invoke(prompt)
    depth_feedback = result.feedback
    depth_score = result.score
    return {
        'depth_feedback': depth_feedback,
        'individual_scores': [depth_score]
    }

@traceable(name="final_evaluation")
def final_evaluation(state: EssayEvaluationState) -> EssayEvaluationState:
    prompt = f"""Based on the following feedbacks, provide a summarized feedback for the essay:
                Clarity Feedback: {state['clarity_feedback']}
                Language Feedback: {state['language_feedback']}
                Depth Feedback: {state['depth_feedback']}
                """
    result = model.invoke(prompt)
    overall_feedback = result.content


    average_score = sum(state['individual_scores']) / len(state['individual_scores'])
    return {
        'overall_feedback': overall_feedback,
        'average_score': average_score
    }


# ----------- Graph Construction -----------
graph = StateGraph(EssayEvaluationState)


graph.add_node('evaluate_clarity',evaluate_clarity)
graph.add_node('evaluate_language',evaluate_language)
graph.add_node('evaluate_depth',evaluate_depth)
graph.add_node('final_evaluation',final_evaluation)


graph.add_edge(START, 'evaluate_clarity')
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_depth')
graph.add_edge('evaluate_clarity', 'final_evaluation')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_depth', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()


# ---------- Direct invoke without wrapper ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay},
        config={
            "run_name": "evaluate_essay_langgraph",
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay),
                "model": "gemini-2.5-flash",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("average_score", 0.0))