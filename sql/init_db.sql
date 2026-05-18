-- CCTV System Database Schema
CREATE DATABASE IF NOT EXISTS cctv_surveillance CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cctv_surveillance;

CREATE TABLE IF NOT EXISTS cameras (
    id INT PRIMARY KEY AUTO_INCREMENT,
    camera_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100),
    location VARCHAR(200),
    rtsp_url VARCHAR(500),
    enabled BOOLEAN DEFAULT TRUE,
    processing_fps INT DEFAULT 15,
    confidence_threshold FLOAT DEFAULT 0.75,
    track_max_age INT DEFAULT 30,
    track_min_hits INT DEFAULT 3,
    track_max_iou_distance FLOAT DEFAULT 0.7,
    track_max_cosine_distance FLOAT DEFAULT 0.4,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_camera_id (camera_id),
    INDEX idx_enabled (enabled)
);

CREATE TABLE IF NOT EXISTS known_faces (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    employee_id VARCHAR(50),
    department VARCHAR(100),
    description TEXT,
    face_embedding BLOB,
    total_detections INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_employee_id (employee_id)
);

CREATE TABLE IF NOT EXISTS known_face_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id INT NOT NULL,
    image_path VARCHAR(500),
    face_embedding BLOB,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES known_faces(id) ON DELETE CASCADE,
    INDEX idx_person_id (person_id)
);

CREATE TABLE IF NOT EXISTS unknown_faces (
    id INT PRIMARY KEY AUTO_INCREMENT,
    track_id VARCHAR(50),
    best_image_path VARCHAR(500),
    face_embedding BLOB,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    appearance_count INT DEFAULT 1,
    camera_id VARCHAR(50),
    highest_confidence FLOAT,
    marked_as_known BOOLEAN DEFAULT FALSE,
    marked_as_person_id INT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_track_camera (track_id, camera_id),
    INDEX idx_first_seen (first_seen),
    INDEX idx_last_seen (last_seen),
    INDEX idx_appearance_count (appearance_count),
    INDEX idx_camera_id (camera_id),
    INDEX idx_marked_as_known (marked_as_known),
    FOREIGN KEY (marked_as_person_id) REFERENCES known_faces(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS unknown_face_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    unknown_id INT NOT NULL,
    image_path VARCHAR(500),
    camera_id VARCHAR(50),
    detected_at TIMESTAMP,
    confidence FLOAT,
    FOREIGN KEY (unknown_id) REFERENCES unknown_faces(id) ON DELETE CASCADE,
    INDEX idx_unknown_id (unknown_id),
    INDEX idx_detected_at (detected_at)
);

CREATE TABLE IF NOT EXISTS detections (
    id INT PRIMARY KEY AUTO_INCREMENT,
    camera_id VARCHAR(50),
    track_id VARCHAR(50),
    person_id INT,
    unknown_id INT,
    person_name VARCHAR(100),
    confidence FLOAT,
    bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
    detected_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES known_faces(id) ON DELETE SET NULL,
    FOREIGN KEY (unknown_id) REFERENCES unknown_faces(id) ON DELETE SET NULL,
    INDEX idx_camera_id (camera_id),
    INDEX idx_track_id (track_id),
    INDEX idx_detected_at (detected_at),
    INDEX idx_person_id (person_id),
    INDEX idx_unknown_id (unknown_id)
);

CREATE TABLE IF NOT EXISTS system_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    event_type VARCHAR(50),
    camera_id VARCHAR(50),
    description TEXT,
    details JSON,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_created_at (created_at)
);
