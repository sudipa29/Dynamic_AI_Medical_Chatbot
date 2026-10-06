from app.gpt_client import generate_gpt_response


# ============================================================
# PREDEFINED RESPONSES
# ============================================================

INTENT_RESPONSES = {

    "greeting": (
        "Hello! Welcome to our support assistant. "
        "How can I help you today?"
    ),

    "goodbye": (
        "Goodbye! Thank you for contacting us. "
        "Have a great day!"
    ),

    "thank_you": (
        "You're welcome! I'm happy to help."
    ),

    "order_status": (
        "Sure! I can help you check your order status. "
        "Please provide your order ID so I can help you further."
    ),

    "order_cancel": (
        "I can help you with order cancellation. "
        "Please provide your order ID so we can check whether "
        "the order can still be cancelled."
    ),

    "refund": (
        "I can help you with a refund. "
        "Please provide your order ID and the reason for the refund."
    ),

    "payment_failed": (
        "I'm sorry that your payment failed. "
        "Please check your card details, available balance, "
        "and try the payment again. If the problem continues, "
        "please contact your payment provider or support team."
    ),

    "payment_issue": (
        "I can help with your payment-related issue. "
        "Please provide the transaction or order details "
        "so we can understand the problem."
    ),

    "delivery": (
        "Delivery time depends on your location and shipping method. "
        "Please provide your order ID so we can check the expected "
        "delivery date."
    ),

    "product_info": (
        "Sure! I can help you with product information. "
        "Please tell me the product name or the details "
        "you would like to know."
    ),

    "account_issue": (
        "I can help you with your account issue. "
        "If you cannot log in, you can try resetting your password. "
        "If the problem continues, please contact support."
    ),

    "order_return": (
        "I can help you with a product return. "
        "Please provide your order ID and the reason for the return."
    ),

    "complaint": (
        "I'm sorry to hear that you had a bad experience. "
        "Please describe the problem so we can help resolve it."
    ),

    "feedback": (
        "Thank you for your feedback! "
        "We appreciate your suggestions and will use them "
        "to improve our service."
    )
}


# ============================================================
# GPT FALLBACK INSTRUCTIONS
# ============================================================

GPT_FALLBACK_CONTEXT = """
The intent classifier could not confidently identify the user's
message as one of the supported customer-service intents.

You are the fallback AI assistant.

Answer the user's question clearly, politely, and concisely.

You are primarily a customer-support assistant.

You may answer general questions when appropriate, but do not
invent:
- order information
- payment information
- refund status
- delivery status
- account information
- customer information

If the user asks for information about a specific order or
account and the required information is not available, ask the
user to provide the relevant details.

If the question is unrelated to customer support, politely explain
that you are primarily designed to assist with customer support.

Do not claim that you have access to systems or information that
has not been provided.
"""


# ============================================================
# GPT FALLBACK RESPONSE
# ============================================================

def generate_fallback_response(user_message, conversation_history=""):
    """
    Generate a GPT response for messages classified as unknown.
    """

    context = GPT_FALLBACK_CONTEXT

    if conversation_history:
        context += (
            "\n\nUse the following context when it is relevant:\n"
            + conversation_history
        )

    return generate_gpt_response(
        user_message=(
            context
            + "\n\nUser message:\n"
            + user_message
        ),
        conversation_history=conversation_history
    )


# ============================================================
# MAIN RESPONSE GENERATION FUNCTION
# ============================================================

def generate_response(
    user_message,
    intent,
    conversation_history=""
):
    """
    Generate a chatbot response based on the detected intent.

    Known intents:
        Use predefined responses.

    Unknown intent:
        Use the Gemini-powered GPT fallback.
    """

    # --------------------------------------------------------
    # Known intent → predefined response
    # --------------------------------------------------------

    if intent in INTENT_RESPONSES:

        return INTENT_RESPONSES[intent]

    # --------------------------------------------------------
    # Unknown intent → GPT
    # --------------------------------------------------------

    if intent == "unknown":

        return generate_fallback_response(
            user_message=user_message,
            conversation_history=conversation_history
        )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    return (
        "I'm sorry, I couldn't understand your request. "
        "Could you please explain it in another way?"
    )


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RESPONSE GENERATOR TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Test predefined responses
    # --------------------------------------------------------

    known_intent_tests = [

        (
            "Hello there",
            "greeting"
        ),

        (
            "Where is my package?",
            "order_status"
        ),

        (
            "I want to cancel my order",
            "order_cancel"
        ),

        (
            "I need my money back",
            "refund"
        ),

        (
            "My card payment failed",
            "payment_failed"
        ),

        (
            "Thank you for your help",
            "thank_you"
        ),

        (
            "I want to return my product",
            "order_return"
        ),

        (
            "I have a complaint",
            "complaint"
        ),

        (
            "I have a suggestion",
            "feedback"
        )
    ]


    print("\n" + "=" * 60)
    print("KNOWN INTENT TESTS")
    print("=" * 60)


    for message, intent in known_intent_tests:

        print("\n" + "-" * 60)

        print(f"User Message: {message}")

        print(f"Detected Intent: {intent}")

        try:

            response = generate_response(
                user_message=message,
                intent=intent
            )

            print(f"Bot Response: {response}")

        except Exception as e:

            print(f"Response generation failed: {e}")


    # --------------------------------------------------------
    # Test GPT fallback
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GPT FALLBACK TEST")
    print("=" * 60)


    unknown_message = "What is quantum physics?"

    print(f"\nUser Message: {unknown_message}")

    print("Detected Intent: unknown")

    try:

        response = generate_response(
            user_message=unknown_message,
            intent="unknown"
        )

        print(f"Bot Response: {response}")

    except Exception as e:

        print("\nGPT fallback request failed.")

        print(f"Error: {e}")


    print("\n" + "=" * 60)
    print("RESPONSE GENERATOR TEST COMPLETED")
    print("=" * 60)