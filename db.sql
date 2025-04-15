CREATE SCHEMA IF NOT EXISTS efficient_study_database;

SET search_path TO efficient_study_database;

CREATE TABLE users(
    id SERIAL PRIMARY KEY, -- Help from ChatGPT
    username VARCHAR(500)
);

CREATE TABLE chatbots(
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Help from ChatGPT
);

CREATE TABLE conversations(
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL, -- Help from ChatGPT
    chatbot_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id), -- Help from ChatGPT
    FOREIGN KEY (chatbot_id) REFERENCES chatbots(id)
);

CREATE TABLE chatbot_messages(
    id SERIAL PRIMARY KEY,
    chatbot_id INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    message TEXT NOT NULL, -- Help from ChatGPT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chatbot_id) REFERENCES chatbots(id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE user_messages(
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);