from app.chatbot_pipeline import process_message


def main():
    conversation_history = ""

    questions = [
        "What is asthma?",
        "What are its symptoms?",
        "How can it be managed?",
        "What causes it?",
    ]

    for question in questions:
        result = process_message(
            user_message=question,
            conversation_history=conversation_history,
        )

        response = str(result.get("response", ""))

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("RESPONSE:", repr(response))
        print("Contains 'atwork':", "atwork" in response)
        print("Contains 'andshould':", "andshould" in response)
        print("Contains 'bedetermined':", "bedetermined" in response)

        conversation_history += (
            "\nUser: " + question
            + "\nAssistant: " + response
            + "\n"
        )


if __name__ == "__main__":
    main()