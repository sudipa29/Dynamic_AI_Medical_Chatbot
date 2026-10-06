class ConversationMemory:

    def __init__(self):
        self.messages = []

    def add_message(self, user_message, bot_response):

        self.messages.append({
            "user": user_message,
            "bot": bot_response
        })

    def get_history(self):

        return self.messages

    def get_formatted_history(self):

        if not self.messages:
            return "No previous conversation."

        history = []

        for message in self.messages:

            history.append(
                f"User: {message['user']}"
            )

            history.append(
                f"Bot: {message['bot']}"
            )

        return "\n".join(history)

    def clear(self):

        self.messages = []


if __name__ == "__main__":

    print("=" * 60)
    print("CONVERSATION MEMORY TEST")
    print("=" * 60)

    memory = ConversationMemory()

    memory.add_message(
        "I want a refund.",
        "You can request a refund within 7 days."
    )

    memory.add_message(
        "My order number is ORD12345.",
        "Thank you. I have noted your order number."
    )

    print("\nConversation History:")

    print(
        memory.get_formatted_history()
    )

    print("\n" + "=" * 60)
    print("MEMORY TEST COMPLETED")
    print("=" * 60)