
from app.intent_detection import predict_intent_details
from app.response_generator import generate_response


# ============================================================
# 1. CHATBOT PIPELINE
# ============================================================

def process_message(user_message, conversation_history=""):
    """
    Process one user message through the complete chatbot pipeline.

    Steps:
    1. Validate the message.
    2. Detect the user's intent.
    3. Apply confidence and fallback rules.
    4. Generate a response.
    5. Return the intent, confidence, and response.
    """

    # --------------------------------------------------------
    # STEP 1: Validate user input
    # --------------------------------------------------------

    if not user_message or not user_message.strip():

        return {
            "user_message": user_message,
            "intent": "unknown",
            "confidence": 0.0,
            "second_intent": "unknown",
            "second_confidence": 0.0,
            "confidence_margin": 0.0,
            "response": "Please enter a message so I can help you."
        }

    user_message = user_message.strip()

    # --------------------------------------------------------
    # STEP 2: Detect intent
    # --------------------------------------------------------

    (
        detected_intent,
        confidence,
        second_intent,
        second_confidence,
        confidence_margin
    ) = predict_intent_details(user_message)

    # --------------------------------------------------------
    # STEP 3: Generate response
    # --------------------------------------------------------

    response = generate_response(
        user_message=user_message,
        intent=detected_intent,
        conversation_history=conversation_history
    )

    # --------------------------------------------------------
    # STEP 4: Return the complete result
    # --------------------------------------------------------

    return {
        "user_message": user_message,
        "intent": detected_intent,
        "confidence": confidence,
        "second_intent": second_intent,
        "second_confidence": second_confidence,
        "confidence_margin": confidence_margin,
        "response": response
    }


# ============================================================
# 2. INTERACTIVE CHATBOT
# ============================================================

def start_chatbot():

    print("\n" + "=" * 60)
    print("DYNAMIC AI CHATBOT (GPT-POWERED)")
    print("=" * 60)

    print("\nHello! Welcome to our customer support chatbot.")
    print("Ask me about orders, payments, refunds, and more.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("\n" + "-" * 60)

    # Simple conversation history for the current session
    conversation_history = ""

    while True:

        # ----------------------------------------------------
        # STEP 1: Get user message
        # ----------------------------------------------------

        user_message = input("\nYou: ").strip()

        # ----------------------------------------------------
        # STEP 2: Exit condition
        # ----------------------------------------------------

        if user_message.lower() in ["exit", "quit", "bye"]:

            print("\nBot: Goodbye! Thank you for chatting with me.")
            print("Have a great day!")

            break

        # ----------------------------------------------------
        # STEP 3: Process the message
        # ----------------------------------------------------

        try:

            result = process_message(
                user_message=user_message,
                conversation_history=conversation_history
            )

            # ------------------------------------------------
            # STEP 4: Display intent information
            # ------------------------------------------------

            print("\n--- Intent Analysis ---")

            print(f"Detected Intent: {result['intent']}")

            print(
                f"Top Confidence: "
                f"{result['confidence']:.2%}"
            )

            print(
                f"Second Intent: "
                f"{result['second_intent']}"
            )

            print(
                f"Second Confidence: "
                f"{result['second_confidence']:.2%}"
            )

            print(
                f"Confidence Margin: "
                f"{result['confidence_margin']:.2%}"
            )

            # ------------------------------------------------
            # STEP 5: Display chatbot response
            # ------------------------------------------------

            print("\nBot:", result["response"])

            # ------------------------------------------------
            # STEP 6: Save session conversation history
            # ------------------------------------------------

            conversation_history += (
                f"\nUser: {user_message}"
                f"\nBot: {result['response']}\n"
            )

        except Exception as error:

            print("\nBot: Sorry, something went wrong.")

            print(
                "Please check the error below "
                "and try again."
            )

            print(f"Error: {error}")


# ============================================================
# 3. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    start_chatbot()