
import mysql.connector


# ============================================================
# MYSQL DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "ai_chatbot"
}


# ============================================================
# GET MYSQL CONNECTION
# ============================================================

def get_connection():

    connection = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )

    return connection


# ============================================================
# CREATE USER
# ============================================================

def create_user(name, email):

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO users (
            name,
            email
        )
        VALUES (%s, %s)
    """

    cursor.execute(
        query,
        (
            name,
            email
        )
    )

    connection.commit()

    user_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return user_id


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(user_id):

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO conversations (
            user_id
        )
        VALUES (%s)
    """

    cursor.execute(
        query,
        (
            user_id,
        )
    )

    connection.commit()

    conversation_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return conversation_id


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id,
    sender,
    message,
    intent=None,
    sentiment=None,
    response_time=None
):

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO messages (
            conversation_id,
            sender,
            message,
            intent,
            sentiment,
            response_time
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            conversation_id,
            sender,
            message,
            intent,
            sentiment,
            response_time
        )
    )

    connection.commit()

    message_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return message_id


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================

def get_conversation_history(conversation_id):

    connection = get_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            message_id,
            sender,
            message,
            intent,
            sentiment,
            response_time,
            created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY created_at ASC, message_id ASC
    """

    cursor.execute(
        query,
        (
            conversation_id,
        )
    )

    history = cursor.fetchall()

    cursor.close()
    connection.close()

    return history


# ============================================================
# GET FORMATTED CONVERSATION HISTORY
# ============================================================

def get_formatted_conversation_history(
    conversation_id
):

    history = get_conversation_history(
        conversation_id
    )

    if not history:

        return "No previous conversation."


    formatted_history = []


    for message in history:

        if message["sender"] == "user":

            formatted_history.append(
                f"User: {message['message']}"
            )

        elif message["sender"] == "bot":

            formatted_history.append(
                f"Bot: {message['message']}"
            )


    return "\n".join(
        formatted_history
    )


# ============================================================
# SAVE FEEDBACK
# ============================================================

def save_feedback(
    message_id,
    rating,
    feedback_text=None
):

    connection = get_connection()

    cursor = connection.cursor()

    query = """
        INSERT INTO feedback (
            message_id,
            rating,
            feedback_text
        )
        VALUES (%s, %s, %s)
    """

    cursor.execute(
        query,
        (
            message_id,
            rating,
            feedback_text
        )
    )

    connection.commit()

    feedback_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return feedback_id


# ============================================================
# GET FEEDBACK SUMMARY
# ============================================================

def get_feedback_summary():

    connection = get_connection()

    cursor = connection.cursor(
        dictionary=True
    )


    # --------------------------------------------------------
    # TOTAL FEEDBACK + AVERAGE RATING
    # --------------------------------------------------------

    query = """
        SELECT
            COUNT(*) AS total_feedback,
            AVG(rating) AS average_rating
        FROM feedback
    """

    cursor.execute(query)

    summary = cursor.fetchone()


    # --------------------------------------------------------
    # POSITIVE FEEDBACK
    # Rating 4 or 5
    # --------------------------------------------------------

    query = """
        SELECT
            COUNT(*) AS positive_feedback
        FROM feedback
        WHERE rating >= 4
    """

    cursor.execute(query)

    positive_result = cursor.fetchone()


    # --------------------------------------------------------
    # NEGATIVE FEEDBACK
    # Rating 1, 2, or 3
    # --------------------------------------------------------

    query = """
        SELECT
            COUNT(*) AS negative_feedback
        FROM feedback
        WHERE rating <= 3
    """

    cursor.execute(query)

    negative_result = cursor.fetchone()


    cursor.close()
    connection.close()


    return {
        "total_feedback": summary["total_feedback"],
        "average_rating": summary["average_rating"],
        "positive_feedback": positive_result[
            "positive_feedback"
        ],
        "negative_feedback": negative_result[
            "negative_feedback"
        ]
    }


# ============================================================
# GET FEEDBACK DISTRIBUTION
# ============================================================

def get_feedback_distribution():

    connection = get_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    query = """
        SELECT
            rating,
            COUNT(*) AS count
        FROM feedback
        GROUP BY rating
        ORDER BY rating
    """

    cursor.execute(query)

    distribution = cursor.fetchall()

    cursor.close()
    connection.close()

    return distribution


# ============================================================
# DATABASE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "MYSQL DATABASE + FEEDBACK ANALYTICS TEST"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # CONVERSATION HISTORY TEST
    # --------------------------------------------------------

    try:

        conversation_id = 2

        print(
            f"\nConversation ID: {conversation_id}"
        )


        formatted_history = (
            get_formatted_conversation_history(
                conversation_id
            )
        )


        print(
            "\nFormatted Conversation History:"
        )

        print(
            "----------------------------------------"
        )

        print(
            formatted_history
        )


    except mysql.connector.Error as error:

        print(
            "\nMySQL operation failed."
        )

        print(
            f"Error: {error}"
        )


    # --------------------------------------------------------
    # FEEDBACK SUMMARY TEST
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FEEDBACK SUMMARY"
    )

    print(
        "=" * 60
    )


    try:

        summary = get_feedback_summary()


        print(
            f"\nTotal Feedback: "
            f"{summary['total_feedback']}"
        )


        if summary["average_rating"] is not None:

            print(
                f"Average Rating: "
                f"{summary['average_rating']:.2f}"
            )

        else:

            print(
                "Average Rating: 0.00"
            )


        print(
            f"Positive Feedback: "
            f"{summary['positive_feedback']}"
        )


        print(
            f"Negative Feedback: "
            f"{summary['negative_feedback']}"
        )


    except mysql.connector.Error as error:

        print(
            "\nFeedback summary failed."
        )

        print(
            f"Error: {error}"
        )


    # --------------------------------------------------------
    # FEEDBACK DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FEEDBACK DISTRIBUTION"
    )

    print(
        "=" * 60
    )


    try:

        distribution = (
            get_feedback_distribution()
        )


        if distribution:

            for row in distribution:

                print(
                    f"Rating {row['rating']}: "
                    f"{row['count']}"
                )

        else:

            print(
                "No feedback available."
            )


    except mysql.connector.Error as error:

        print(
            "\nFeedback distribution failed."
        )

        print(
            f"Error: {error}"
        )


    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FEEDBACK ANALYTICS TEST COMPLETED"
    )

    print(
        "=" * 60
    )