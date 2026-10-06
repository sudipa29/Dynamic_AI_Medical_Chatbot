CREATE DATABASE ai_chatbot;
USE ai_chatbot;
show databases;
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(150),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
    conversation_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE messages (
    message_id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    sender VARCHAR(20),
    message TEXT,
    intent VARCHAR(100),
    sentiment VARCHAR(30),
    response_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE feedback (
    feedback_id INT AUTO_INCREMENT PRIMARY KEY,
    message_id INT,
    rating INT,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);

SHOW TABLES;
DESCRIBE users;
DESCRIBE conversations;
DESCRIBE messages;
DESCRIBE feedback;

SELECT * FROM users;
SELECT * FROM conversations;
SELECT * FROM messages;
SELECT * FROM feedback;

SELECT
    feedback_id,
    message_id,
    rating,
    feedback_text,
    created_at
FROM feedback
ORDER BY feedback_id DESC;

SELECT
    f.feedback_id,
    f.message_id,
    f.rating,
    f.feedback_text,
    f.created_at
FROM feedback f
ORDER BY f.feedback_id;

INSERT INTO feedback (
    message_id,
    rating,
    feedback_text
)
VALUES (
    22,
    2,
    'The response was not very helpful.'
);

SELECT
    feedback_id,
    message_id,
    rating,
    feedback_text,
    created_at
FROM feedback
ORDER BY feedback_id;