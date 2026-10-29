"""Cover letter, outreach message, and resume title generation service."""

import json
import logging
from typing import Any

from app.ai_limits import MAX_JOB_CHARACTERS, validate_source_size
from app.config import load_config_file
from app.llm import complete, complete_json
from app.prompts import get_language_name
from app.prompts.templates import (
    COVER_LETTER_PROMPT,
    GENERATE_TITLE_PROMPT,
    OUTREACH_MESSAGE_PROMPT,
)


def _resolve_feature_prompt(
    custom_key: str,
    default_template: str,
) -> tuple[str, bool]:
    """Resolve a feature-prompt template at runtime.

    Returns ``(template, is_custom)``. If the stored custom prompt is
    empty or absent, returns the default template. The ``is_custom`` flag
    lets callers decide whether to fall back to the default on a format
    failure (defensive — save-time validation should have caught a
    malformed custom prompt).
    """
    stored = load_config_file()
    custom = (stored.get(custom_key) or "").strip()
    if not custom:
        return default_template, False
    return custom, True


async def generate_cover_letter(
    resume_data: dict[str, Any],
    job_description: str,
    language: str = "en",
) -> str:
    """Generate a cover letter based on resume and job description.

    Args:
        resume_data: Structured resume data (ResumeData format)
        job_description: Target job description text
        language: Output language code (en, es, zh, ja)

    Returns:
        Generated cover letter as plain text
    """
    validate_source_size(job_description, MAX_JOB_CHARACTERS)
    output_language = get_language_name(language)

    template, is_custom = _resolve_feature_prompt(
        "cover_letter_prompt", COVER_LETTER_PROMPT
    )
    try:
        prompt = template.format(
            job_description=job_description,
            resume_data=json.dumps(resume_data),
            output_language=output_language,
        )
    except (KeyError, IndexError, ValueError) as e:
        # str.format() raises KeyError for unknown placeholders, IndexError for
        # positional out-of-range, and ValueError for unmatched/invalid braces
        # (e.g., ``{foo``). If the failing template is the built-in default,
        # something is broken upstream and the caller should see it — re-raise.
        # If it's a user-supplied custom prompt, fall back to the default with a
        # warning so generation doesn't crash on out-of-band disk edits.
        if not is_custom:
            raise
        logging.warning(
            "Custom cover letter prompt failed to format (%s); falling back to default",
            e,
        )
        prompt = COVER_LETTER_PROMPT.format(
            job_description=job_description,
            resume_data=json.dumps(resume_data),
            output_language=output_language,
        )

    result = await complete(
        prompt=prompt,
        system_prompt="You are a professional career coach and resume writer. Write compelling, personalized cover letters.",
        max_tokens=2048,
    )

    return result.strip()


async def generate_outreach_message(
    resume_data: dict[str, Any],
    job_description: str,
    language: str = "en",
) -> str:
    """Generate a cold outreach message for networking.

    Args:
        resume_data: Structured resume data (ResumeData format)
        job_description: Target job description text
        language: Output language code (en, es, zh, ja)

    Returns:
        Generated outreach message as plain text
    """
    validate_source_size(job_description, MAX_JOB_CHARACTERS)
    output_language = get_language_name(language)

    template, is_custom = _resolve_feature_prompt(
        "outreach_message_prompt", OUTREACH_MESSAGE_PROMPT
    )
    try:
        prompt = template.format(
            job_description=job_description,
            resume_data=json.dumps(resume_data),
            output_language=output_language,
        )
    except (KeyError, IndexError, ValueError) as e:
        # See generate_cover_letter for rationale on the exception set.
        if not is_custom:
            raise
        logging.warning(
            "Custom outreach message prompt failed to format (%s); falling back to default",
            e,
        )
        prompt = OUTREACH_MESSAGE_PROMPT.format(
            job_description=job_description,
            resume_data=json.dumps(resume_data),
            output_language=output_language,
        )

    result = await complete(
        prompt=prompt,
        system_prompt="You are a professional networking coach. Write genuine, engaging cold outreach messages.",
        max_tokens=1024,
    )

    return result.strip()


async def generate_resume_title(
    job_description: str,
    language: str = "en",
) -> str:
    """Generate a short descriptive title from a job description.

    Args:
        job_description: Target job description text
        language: Output language code (en, es, zh, ja)

    Returns:
        Generated title like "Senior Frontend Engineer @ Stripe"
    """
    validate_source_size(job_description, MAX_JOB_CHARACTERS)
    output_language = get_language_name(language)

    prompt = GENERATE_TITLE_PROMPT.format(
        job_description=job_description,
        output_language=output_language,
    )

    result = await complete(
        prompt=prompt,
        system_prompt="You extract job titles and company names from job descriptions.",
        max_tokens=500,
        temperature=0.3,
    )

    # Strip quotes and whitespace, truncate to 80 chars
    title = result.strip().strip("\"'")
    return title[:80]


async def translate_cover_letter(
    content: str,
    target_language: str = "id",
) -> str:
    """Translate cover letter into target language (default: Indonesian)."""
    lang_name = get_language_name(target_language)
    prompt = f"""You are an expert bilingual professional business writer.
Translate the following cover letter into {lang_name} (Bahasa Indonesia).

Requirements:
- Use formal, professional Indonesian business correspondence standards (e.g. greeting: "Yth. Manajer Perekrutan," or named recipient if provided; sign-off: "Hormat saya,").
- Keep all facts, achievements, metrics, company names, dates, and technical terms accurate.
- Maintain a confident, polite, and persuasive tone that feels completely natural to a native Indonesian professional.
- Preserve paragraph breaks and structure.
- Do NOT add markdown code fences, quotes, or em dashes ("—").
- Output the translated cover letter text only, nothing else.

Cover Letter to Translate:
{content}
"""
    result = await complete(
        prompt=prompt,
        system_prompt="You are a professional executive resume writer and translator specializing in Indonesian business correspondence.",
        max_tokens=2048,
    )
    return result.strip()


async def translate_resume_data(
    resume_data: dict[str, Any],
    target_language: str = "id",
) -> dict[str, Any]:
    """Translate resume content fields into target language (default: Indonesian)."""
    lang_name = get_language_name(target_language)
    prompt = f"""You are an expert bilingual resume translator. Translate the descriptive text fields of this resume into {lang_name} (Bahasa Indonesia).

Instructions:
1. Translate these fields into professional, fluent Indonesian:
   - 'summary': Translate into professional Indonesian business summary.
   - 'workExperience': In each entry, translate the bullet strings in 'description'.
   - 'personalProjects': Translate the bullet strings in 'description'.
   - 'education': Translate 'degree' and 'fieldOfStudy' into standard Indonesian equivalents (e.g. "Bachelor of Computer Science" -> "Sarjana Komputer", etc.).
2. STRICTLY PRESERVE (DO NOT CHANGE):
   - All JSON keys and schema structure must remain EXACTLY identical.
   - Keep names, emails, phone numbers, addresses/locations, links unchanged.
   - Keep company names, school names, dates, metrics, percentages, tool and programming language names unchanged.
3. Return ONLY valid JSON matching the input schema.

Resume Data:
{json.dumps(resume_data)}
"""
    result = await complete_json(
        prompt=prompt,
        system_prompt="You are a professional resume writer and translator.",
    )
    if isinstance(result, dict):
        return result
    return resume_data
