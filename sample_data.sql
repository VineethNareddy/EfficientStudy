CREATE SCHEMA IF NOT EXISTS efficient_study_database;

SET search_path TO efficient_study_database;

CREATE OR REPLACE FUNCTION 

--% We will insert 4 users and 4 chatbots with sample conversation messages

INSERT INTO users (username)
VALUES ("Goob");

INSERT INTO chatbots DEFAULT VALUES;

