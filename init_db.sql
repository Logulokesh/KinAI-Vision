-- Project 1 Tables
CREATE TABLE IF NOT EXISTS event_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('SINGLE_KNOWN', 'FAMILY_PROFILE', 'UNKNOWN_WITH_KNOWN', 'SUSPECT', 'NO_DETECTION')),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS family_member (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS system_state (
    id SERIAL PRIMARY KEY,
    no_detection_count INTEGER DEFAULT 0 NOT NULL,
    no_detection_flag INTEGER DEFAULT 0 NOT NULL
);

CREATE TABLE IF NOT EXISTS music_schedule (
    id SERIAL PRIMARY KEY,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    playlist_id VARCHAR(255) NOT NULL
);

-- Project 2 Tables
CREATE TABLE IF NOT EXISTS faces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding BYTEA NOT NULL,
    last_updated VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS known_persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding BYTEA NOT NULL,
    first_seen VARCHAR(50) NOT NULL,
    last_seen VARCHAR(50) NOT NULL,
    detection_count INTEGER DEFAULT 1 NOT NULL,
    reference_image TEXT
);

CREATE TABLE IF NOT EXISTS unknown_visitors (
    ulid VARCHAR(50) PRIMARY KEY,
    embedding BYTEA NOT NULL,
    first_seen VARCHAR(50) NOT NULL,
    last_seen VARCHAR(50) NOT NULL,
    visit_count INTEGER DEFAULT 1 NOT NULL,
    camera_id VARCHAR(50),
    image_path TEXT
);

CREATE TABLE IF NOT EXISTS family_profiles (
    profile_id SERIAL PRIMARY KEY,
    profile_name VARCHAR(255) NOT NULL,
    member_ids TEXT NOT NULL,
    created_at VARCHAR(50) NOT NULL,
    last_updated VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    timestamp VARCHAR(50) NOT NULL,
    device VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    image_path TEXT,
    camera_id VARCHAR(50),
    unknown_id VARCHAR(50)
);