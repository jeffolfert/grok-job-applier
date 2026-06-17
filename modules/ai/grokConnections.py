''' 
Author:     Enhanced for xAI Grok by Grok (based on original GodsScion project)
LinkedIn:   N/A
GitHub:     Improvements on https://github.com/GodsScion/Auto_job_applier_linkedIn

License:    GNU Affero General Public License
            https://www.gnu.org/licenses/agpl-3.0.en.html
'''

"""
Grok (xAI) AI Connections Module

This module provides Grok-specific client creation and AI functions (extract skills, answer questions)
using the xAI API, which is OpenAI-compatible.

Usage in secrets.py:
  ai_provider = "grok"
  llm_api_url = "https://api.x.ai/v1/"
  llm_api_key = "xai-YOUR_KEY_HERE"
  llm_model = "grok-3"   # or "grok-4", "grok-beta", etc. Check https://x.ai/api for current models
  llm_spec = "openai"

Grok excels at strong reasoning, helpfulness, and less censored/more truthful responses — great for job fit analysis and high-quality tailoring.
"""

from config.secrets import *
from config.settings import showAiErrorAlerts
from config.personals import ethnicity, gender, disability_status, veteran_status
from config.questions import *
from config.search import security_clearance, did_masters

from modules.helpers import print_lg, critical_error_log, convert_to_json
from modules.ai.prompts import *

from pyautogui import confirm
from openai import OpenAI
from openai.types.model import Model
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from typing import Iterator, Literal


apiCheckInstructions = """

1. Make sure your xAI API key is valid and has credits.
2. Confirm the base URL is exactly "https://api.x.ai/v1/" (with trailing slash).
3. Check that your chosen llm_model is available (e.g. grok-3, grok-4).
4. Open `secrets.py` in `/config` folder to configure.

ERROR:
"""

def ai_error_alert(message: str, stackTrace: str, title: str = "Grok (xAI) Connection Error") -> None:
    """
    Function to show an AI error alert and log it.
    """
    global showAiErrorAlerts
    if showAiErrorAlerts:
        if "Pause AI error alerts" == confirm(f"{message}{stackTrace}\n", title, ["Pause AI error alerts", "Okay Continue"]):
            showAiErrorAlerts = False
    critical_error_log(message, stackTrace)


def ai_check_error(response: ChatCompletion | ChatCompletionChunk) -> None:
    """
    Function to check if an error occurred.
    * Takes in `response` of type `ChatCompletion` or `ChatCompletionChunk`
    * Raises a `ValueError` if an error is found
    """
    if response.model_extra.get("error"):
        raise ValueError(
            f'Error occurred with xAI Grok API: "{response.model_extra.get("error")}"'
        )


def grok_create_client() -> OpenAI:
    """
    Function to create an xAI Grok client (OpenAI compatible).
    * Takes no arguments
    * Returns an `OpenAI` object configured for xAI
    """
    try:
        print_lg("Creating xAI Grok client...")
        if not use_AI:
            raise ValueError("AI is not enabled! Please enable it by setting `use_AI = True` in `secrets.py` in `config` folder.")
        
        # Force correct base for Grok if user didn't set it precisely
        base = llm_api_url if llm_api_url and "x.ai" in llm_api_url.lower() else "https://api.x.ai/v1/"
        
        client = OpenAI(base_url=base, api_key=llm_api_key)

        # Optional: list models (may be limited on xAI)
        try:
            models = client.models.list()
            print_lg("Available models from xAI (sample):")
            print_lg([m.id for m in models.data[:5]] if hasattr(models, 'data') else models)
        except Exception:
            pass  # xAI model listing may be restricted; not critical

        print_lg("---- SUCCESSFULLY CREATED GROK (xAI) CLIENT! ----")
        print_lg(f"Using API URL: {base}")
        print_lg(f"Using Model: {llm_model}")
        print_lg("Grok is excellent for deep reasoning and personalized job application content.")
        print_lg("Check './config/secrets.py' for more details.\n")
        print_lg("---------------------------------------------")

        return client
    except Exception as e:
        ai_error_alert(f"Error occurred while creating Grok (xAI) client. {apiCheckInstructions}", str(e))
        raise


def grok_close_client(client: OpenAI) -> None:
    """
    Function to close a Grok client (same as OpenAI client).
    """
    try:
        if client:
            print_lg("Closing Grok (xAI) client...")
            client.close()
    except Exception as e:
        ai_error_alert("Error occurred while closing Grok client.", str(e))


def grok_completion(client: OpenAI, messages: list[dict], response_format: dict = None, temperature: float = 0.3, stream: bool = stream_output) -> dict | ValueError:
    """
    Chat completion wrapper for Grok.
    Uses slightly higher default temperature for creative tailoring while keeping factual.
    """
    if not client: raise ValueError("Grok client is not available!")

    params = {"model": llm_model, "messages": messages, "stream": stream}

    # Grok models generally support temperature
    params["temperature"] = temperature

    if response_format:
        # Grok supports json_object / json_schema in recent versions
        params["response_format"] = response_format

    completion = client.chat.completions.create(**params)

    result = ""
    
    if stream:
        print_lg("--GROK STREAMING STARTED")
        for chunk in completion:
            ai_check_error(chunk)
            chunkMessage = chunk.choices[0].delta.content
            if chunkMessage != None:
                result += chunkMessage
            print_lg(chunkMessage, end="", flush=True)
        print_lg("\n--GROK STREAMING COMPLETE")
    else:
        ai_check_error(completion)
        result = completion.choices[0].message.content
    
    if response_format:
        result = convert_to_json(result)
    
    print_lg("\nGrok Answer:\n")
    print_lg(result, pretty=response_format)
    return result


def grok_extract_skills(client: OpenAI, job_description: str, stream: bool = stream_output) -> dict | ValueError:
    """
    Extract skills using Grok (reuses the excellent prompt from prompts.py).
    """
    print_lg("-- EXTRACTING SKILLS FROM JOB DESCRIPTION (via Grok)")
    try:        
        prompt = extract_skills_prompt.format(job_description)
        messages = [{"role": "user", "content": prompt}]
        # Use the json schema if supported
        return grok_completion(client, messages, response_format=extract_skills_response_format, stream=stream)
    except Exception as e:
        ai_error_alert(f"Error occurred while extracting skills from job description using Grok. {apiCheckInstructions}", str(e))
        return {"tech_stack": [], "technical_skills": [], "other_skills": [], "required_skills": [], "nice_to_have": []}


def grok_answer_question(
    client: OpenAI, 
    question: str, 
    options: list[str] | None = None, 
    question_type: Literal['text', 'textarea', 'single_select', 'multiple_select'] = 'text',  
    job_description: str | None = None,
    about_company: str | None = None,
    user_information_all: str = ""
) -> str | ValueError:
    """
    Answer a job application question intelligently using Grok.
    """
    print_lg(f"-- ANSWERING QUESTION (Grok): {question}")
    try:
        user_info = user_information_all or f"""
        Years of experience: {years_of_experience}
        Require visa: {require_visa}
        Website: {website}
        LinkedIn: {linkedIn}
        US Citizenship: {us_citizenship}
        Desired salary: {desired_salary}
        Notice period: {notice_period}
        Current CTC: {current_ctc}
        LinkedIn Headline: {linkedin_headline}
        LinkedIn Summary: {linkedin_summary}
        Cover letter (base): {cover_letter}
        Recent employer: {recent_employer}
        """

        prompt = ai_answer_prompt.format(user_info, question)

        if job_description:
            prompt += f"\n\nJOB DESCRIPTION (for context):\n{job_description[:2000]}"
        if about_company:
            prompt += f"\n\nABOUT COMPANY (for context):\n{about_company[:1500]}"

        if options:
            prompt += f"\n\nAVAILABLE OPTIONS: {options}"

        messages = [{"role": "user", "content": prompt}]
        answer = grok_completion(client, messages, temperature=0.4)

        # Basic cleanup
        if isinstance(answer, dict):
            answer = str(answer)

        answer = answer.strip().strip('"').strip("'")

        # Enforce concise for certain types
        if question_type in ['text', 'single_select'] and len(answer) > 120:
            answer = answer[:117] + "..."

        return answer
    except Exception as e:
        ai_error_alert(f"Error occurred while answering question with Grok. {apiCheckInstructions}", str(e))
        return ""


# Optional: Job fit scoring function (can be called from main bot)
def grok_job_fit_score(client: OpenAI, user_profile: str, job_title: str, company: str, job_description: str, location: str = "") -> dict:
    """
    Uses Grok to score how great a fit this job is for the user.
    Returns: {"score": int 0-100, "why_great_fit": str, "risks_gaps": str, "tailoring_advice": str, "recommendation": str}
    """
    print_lg("-- GROK JOB FIT SCORING --")
    try:
        prompt = job_fit_score_prompt.format(
            user_profile=user_profile[:2500],
            job_title=job_title,
            company=company,
            location=location,
            job_description=job_description[:4000]
        )
        messages = [{"role": "user", "content": prompt}]
        result = grok_completion(client, messages, temperature=0.3, response_format={
            "type": "json_object"
        })
        if isinstance(result, str):
            result = convert_to_json(result)
        # Ensure score is int
        if "score" in result:
            try:
                result["score"] = int(result["score"])
            except:
                result["score"] = 60
        return result
    except Exception as e:
        print_lg("Grok fit scoring failed, returning neutral score:", e)
        return {
            "score": 65,
            "why_great_fit": "Unable to fully evaluate with AI at this time.",
            "risks_gaps": "Manual review recommended.",
            "tailoring_advice": "Highlight relevant experience from your background.",
            "recommendation": "Consider applying if it matches your goals."
        }


# ============================================================
# RESUME + COVER LETTER GENERATION (New Feature)
# ============================================================

def grok_tailor_resume(client: OpenAI, user_resume_text: str, job_title: str, company: str, job_description: str) -> dict:
    """
    Uses Grok to generate a highly tailored resume structure for this specific job.
    Returns dict with 'tailored_summary', 'tailored_experience' (list), 'tailored_skills', 'key_highlights'.
    """
    print_lg("-- GROK TAILORING RESUME FOR JOB --")
    try:
        prompt = resume_tailoring_prompt.format(
            user_resume_text=user_resume_text[:4500],
            job_title=job_title,
            company=company,
            job_description=job_description[:4500]
        )
        messages = [{"role": "user", "content": prompt}]
        result = grok_completion(client, messages, temperature=0.25, response_format={"type": "json_object"})
        
        if isinstance(result, str):
            result = convert_to_json(result)
        
        # Basic validation / defaults
        if not isinstance(result, dict):
            result = {}
        result.setdefault("tailored_summary", "Experienced professional with relevant background for this role.")
        result.setdefault("tailored_experience", [])
        result.setdefault("tailored_skills", [])
        result.setdefault("key_highlights", "")
        return result
    except Exception as e:
        ai_error_alert(f"Error generating tailored resume with Grok. {apiCheckInstructions}", str(e))
        return {
            "tailored_summary": "Strong background in relevant technologies and problem-solving.",
            "tailored_experience": [],
            "tailored_skills": [],
            "key_highlights": "Please review and customize manually."
        }


def grok_generate_cover_letter(client: OpenAI, user_background: str, job_title: str, company: str, job_description: str) -> str:
    """
    Uses Grok to generate a personalized cover letter for the job.
    Returns the full cover letter text (ready to paste or save).
    """
    print_lg("-- GROK GENERATING COVER LETTER --")
    try:
        # Create a short requirements summary for the prompt
        requirements_summary = job_description[:1800]
        
        prompt = cover_letter_prompt.format(
            user_background=user_background[:2200],
            job_title=job_title,
            company=company,
            job_requirements_summary=requirements_summary
        )
        messages = [{"role": "user", "content": prompt}]
        letter = grok_completion(client, messages, temperature=0.6)
        
        if isinstance(letter, dict):
            letter = str(letter)
        return letter.strip()
    except Exception as e:
        ai_error_alert(f"Error generating cover letter with Grok. {apiCheckInstructions}", str(e))
        return f"""Dear Hiring Manager,

I am excited to apply for the {job_title} position at {company}. With my background in relevant technologies and proven ability to deliver impact, I am confident I would be a strong addition to your team.

[Please customize this section with 1-2 specific achievements from your background that match the job.]

I would welcome the opportunity to discuss how my skills and experience align with the needs of your team.

Thank you for your consideration.
"""


def generate_tailored_materials(client: OpenAI, base_resume_text: str, job_title: str, company: str, job_description: str, job_id: str = "unknown") -> dict:
    """
    High-level helper: generates both tailored resume data and cover letter.
    Saves files to generated_materials/<job_id>/ and returns paths + text.
    """
    import os
    from modules.resumes.generator import create_resume_docx  # reuse existing basic generator where possible

    os.makedirs(f"generated_materials/{job_id}", exist_ok=True)

    # 1. Get structured tailored resume data
    tailored = grok_tailor_resume(client, base_resume_text, job_title, company, job_description)

    # 2. Generate cover letter text
    cover_text = grok_generate_cover_letter(client, base_resume_text, job_title, company, job_description)

    # 3. Create a PDF resume using the existing generator structure (best effort)
    resume_pdf_path = f"generated_materials/{job_id}/resume_{job_id}.pdf"
    try:
        # Build minimal structured data for the old generator
        user_details = {
            "name": "Tailored Resume",
            "email": "",
            "phone_number": "",
            "address": ""
        }
        summary = tailored.get("tailored_summary", "")
        experience = tailored.get("tailored_experience", [])
        skills = tailored.get("tailored_skills", [])
        
        # The existing generator expects specific keys; we adapt
        create_resume_docx(
            user_details,
            summary,
            experience or [{"company": "See tailored experience", "role": "", "dates": "", "achievements": ""}],
            [],  # projects
            skills,
            []
        )
        # Rename/move the generated files if the function saved them as resume.pdf / resume.docx
        if os.path.exists("resume.pdf"):
            os.rename("resume.pdf", resume_pdf_path)
        if os.path.exists("resume.docx"):
            os.rename("resume.docx", f"generated_materials/{job_id}/resume_{job_id}.docx")
    except Exception as e:
        print_lg("Could not auto-generate PDF resume from structured data (will use text version):", e)
        resume_pdf_path = None

    # Save plain text versions (very useful for Easy Apply textareas)
    summary_path = f"generated_materials/{job_id}/summary_{job_id}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(tailored.get("tailored_summary", ""))

    cover_path = f"generated_materials/{job_id}/cover_letter_{job_id}.txt"
    with open(cover_path, "w", encoding="utf-8") as f:
        f.write(cover_text)

    # Also save full structured data
    import json
    with open(f"generated_materials/{job_id}/tailored_data_{job_id}.json", "w", encoding="utf-8") as f:
        json.dump(tailored, f, indent=2)

    print_lg(f"✅ Generated tailored materials for job {job_id}:")
    print_lg(f"   - Summary: {summary_path}")
    print_lg(f"   - Cover letter: {cover_path}")
    if resume_pdf_path:
        print_lg(f"   - Resume PDF: {resume_pdf_path}")

    return {
        "summary_text": tailored.get("tailored_summary", ""),
        "summary_path": summary_path,
        "cover_letter_text": cover_text,
        "cover_letter_path": cover_path,
        "resume_pdf_path": resume_pdf_path,
        "tailored_data": tailored
    }
