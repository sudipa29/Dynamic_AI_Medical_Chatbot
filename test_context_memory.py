
from app.chatbot_pipeline import process_message


conversation_history = ""

questions = [
    "What is asthma?",
    "What are its symptoms?",
    "How can it be managed?",
    "What causes it?",
]

for question in questions:
    print("\n" + "=" * 50)
    print("USER:", question)

    try:
        result = process_message(
            user_message=question,
            conversation_history=conversation_history,
        )

        print("Intent:", result.get("intent"))
        print("Topic:", result.get("topic"))
        print("Response:", result.get("response"))

        conversation_history += (
            "\nUser: " + question
            + "\nAssistant: " + str(result.get("response", ""))
            + "\n"
        )

    except Exception as error:
        print("ERROR:", repr(error))
        break
