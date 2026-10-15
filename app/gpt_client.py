from openai import OpenAI
from dotenv import load_dotenv
import os
import time


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please check your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30.0,
    max_retries=0
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

# Maximum number of attempts for temporary failures.
# This does NOT apply to daily quota exhaustion.
MAX_RETRIES = 3

# Seconds before retrying temporary failures.
RETRY_DELAYS = [2, 4]


# ============================================================
# RETRY CHECK
# ============================================================

def is_retryable_error(error):
    """
    Determine whether a Gemini error is temporary
    and safe to retry.

    Daily quota exhaustion is NOT retryable.
    """

    error_text = str(error).lower()

    # --------------------------------------------------------
    # DAILY QUOTA EXHAUSTION
    # --------------------------------------------------------
    # Example:
    # GenerateRequestsPerDayPerProjectPerModel-FreeTier
    # RESOURCE_EXHAUSTED
    # retry in 16h...
    # --------------------------------------------------------

    daily_quota_keywords = [
        "generatedrequestsperday",
        "generaterequestsperday",
        "perdayperprojectpermodel",
        "daily quota",
        "quota exceeded for metric",
        "retry in"
    ]

    for keyword in daily_quota_keywords:
        if keyword in error_text:
            return False

    # --------------------------------------------------------
    # TEMPORARY RATE LIMIT
    # --------------------------------------------------------

    temporary_rate_limit_keywords = [
        "429",
        "rate limit",
        "too many requests"
    ]

    for keyword in temporary_rate_limit_keywords:
        if keyword in error_text:
            return True

    # --------------------------------------------------------
    # TEMPORARY SERVER ERRORS
    # --------------------------------------------------------

    temporary_server_keywords = [
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "temporarily unavailable",
        "overloaded",
        "high demand"
    ]

    for keyword in temporary_server_keywords:
        if keyword in error_text:
            return True

    # --------------------------------------------------------
    # TIMEOUT / CONNECTION ERRORS
    # --------------------------------------------------------

    temporary_connection_keywords = [
        "timeout",
        "timed out",
        "connection error",
        "connection reset",
        "temporary failure"
    ]

    for keyword in temporary_connection_keywords:
        if keyword in error_text:
            return True

    return False


# ============================================================
# GEMINI RESPONSE GENERATOR
# ============================================================

def generate_gpt_response(
    user_message,
    conversation_history="",
    system_instruction=""
):
    """
    Generate a response using Gemini through the
    OpenAI-compatible API.

    Parameters
    ----------
    user_message : str
        Current user message.

    conversation_history : str
        Previous conversation context.

    system_instruction : str
        Instructions that control the assistant behavior.

    Returns
    -------
    str
        Gemini-generated response.

    Raises
    ------
    RuntimeError
        If Gemini cannot generate a response.
    """

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt_parts = []

    if system_instruction:
        prompt_parts.append(
            "Assistant instructions:\n"
            + system_instruction
        )

    if conversation_history:
        prompt_parts.append(
            "Conversation history:\n"
            + conversation_history
        )

    prompt_parts.append(
        "Current user message:\n"
        + user_message
    )

    final_prompt = "\n\n".join(prompt_parts)

    last_error = None

    # --------------------------------------------------------
    # GEMINI REQUEST WITH LIMITED RETRIES
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"[Gemini] Sending request "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                timeout=30.0
            )

            # ------------------------------------------------
            # EXTRACT RESPONSE
            # ------------------------------------------------

            if not response.choices:
                raise RuntimeError(
                    "Gemini returned no response choices."
                )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            print("[Gemini] Response generated successfully.")

            return content.strip()

        except Exception as error:

            last_error = error

            print(
                f"[Gemini] Attempt {attempt} failed:"
            )
            print(error)

            # ------------------------------------------------
            # DAILY QUOTA
            # ------------------------------------------------

            error_text = str(error).lower()

            daily_quota = any(
                keyword in error_text
                for keyword in [
                    "generatedrequestsperday",
                    "generaterequestsperday",
                    "perdayperprojectpermodel",
                    "daily quota",
                    "quota exceeded for metric"
                ]
            )

            if daily_quota:

                print(
                    "[Gemini] Daily quota exhausted. "
                    "No retry will be attempted."
                )

                raise RuntimeError(
                    "Gemini daily quota has been exhausted."
                )

            # ------------------------------------------------
            # NON-RETRYABLE ERROR
            # ------------------------------------------------

            if not is_retryable_error(error):

                print(
                    "[Gemini] Error is not retryable."
                )

                raise RuntimeError(
                    f"Gemini request failed: {error}"
                )

            # ------------------------------------------------
            # TEMPORARY ERROR
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                delay = RETRY_DELAYS[attempt - 1]

                print(
                    f"[Gemini] Temporary error. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

    # --------------------------------------------------------
    # ALL ATTEMPTS FAILED
    # --------------------------------------------------------

    raise RuntimeError(
        "Gemini request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GEMINI CLIENT TEST")
    print("=" * 60)

    try:

        result = generate_gpt_response(
            user_message="Explain machine learning in simple words."
        )

        print("\nGemini response:")
        print(result)

    except Exception as error:

        print("\nGemini test failed:")
        print(error)

    print("=" * 60)