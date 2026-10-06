import os

from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please check your .env file."
    )

# Create OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def generate_gpt_response(
    user_message,
    conversation_history=""
):

    system_instruction = """
You are an AI customer support chatbot.

Your responsibilities:
- Help customers with orders, payments, refunds, deliveries, products, and accounts.
- Be polite, professional, and concise.
- Use previous conversation context when it is relevant.
- Do not invent information about orders, refunds, payments, or customer accounts.
- If you need information that is not provided, ask the customer for it.
- If a question is unrelated to customer support, politely explain that you are designed to assist with customer support.
"""

    messages = [
        {
            "role": "system",
            "content": system_instruction
        }
    ]

    if conversation_history:

        messages.append({
            "role": "system",
            "content": (
                "Previous conversation:\n"
                + conversation_history
            )
        })

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model="gemini-3.8-flash",
        messages=messages
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    print("=" * 60)
    print("GPT CUSTOMER SUPPORT TEST")
    print("=" * 60)

    test_messages = [
        "My order is very late.",
        "I need a refund.",
        "My payment failed.",
        "What is the capital of India?"
    ]

    for test_message in test_messages:

        print(f"\nUser: {test_message}")

        try:
            answer = generate_gpt_response(test_message)

            print(f"GPT: {answer}")

        except Exception as e:
            print("\nGPT API request failed.")
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("GPT CUSTOMER SUPPORT TEST COMPLETED")
    print("=" * 60)