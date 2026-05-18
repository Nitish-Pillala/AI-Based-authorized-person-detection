"""Run this once to create the database schema."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mysql.connector
from backend.config import settings

SQL_FILE = os.path.join(os.path.dirname(__file__), "sql", "init_db.sql")


def setup():
    print(f"Connecting to MySQL at {settings.db_host}:{settings.db_port}...")

    conn = mysql.connector.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
    )
    cursor = conn.cursor()

    print("Running database initialization script...")
    with open(SQL_FILE, "r") as f:
        sql = f.read()

    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            try:
                cursor.execute(stmt)
                conn.commit()
            except mysql.connector.Error as e:
                if "already exists" in str(e).lower():
                    pass
                else:
                    print(f"Warning: {e}")

    cursor.close()
    conn.close()
    print("Database setup complete!")
    print(f"\nDatabase: {settings.db_name}")
    print("Tables created: cameras, known_faces, known_face_images,")
    print("                unknown_faces, unknown_face_images, detections, system_logs")


if __name__ == "__main__":
    setup()
