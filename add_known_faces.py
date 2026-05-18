"""
Bulk add known faces to the CCTV system.

Usage:
    python add_known_faces.py

Folder structure expected:
    known_faces_input/
        John_Doe/
            photo1.jpg
            photo2.jpg
        Jane_Smith/
            photo1.jpg
        ...

OR edit the PERSONS list below for manual entry.
"""

import sys
import os
import base64
import httpx
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_BASE = "http://localhost:8000"

# ── Option A: Manual list ─────────────────────────────────────────────────────
# Add entries here: (name, employee_id, department, image_path)
PERSONS = [
    # ("Alice Johnson", "EMP001", "HR",          r"C:\faces\alice.jpg"),
    # ("Bob Smith",     "EMP002", "Engineering", r"C:\faces\bob.jpg"),
]

# ── Option B: Auto-scan a folder ──────────────────────────────────────────────
# Set this path to a folder where each sub-folder = one person's name
INPUT_FOLDER = "./known_faces_input"


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_person(name: str, employee_id: str = None, department: str = None,
               image_path: str = None) -> dict:
    payload = {
        "name": name,
        "employee_id": employee_id,
        "department": department,
    }
    if image_path:
        payload["face_image"] = image_to_base64(image_path)

    resp = httpx.post(f"{API_BASE}/api/known-persons", json=payload, timeout=30)
    return resp.json()


def add_extra_image(person_id: int, image_path: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f.read(), "image/jpeg")}
    resp = httpx.post(f"{API_BASE}/api/known-persons/{person_id}/image",
                      files=files, timeout=30)
    return resp.json()


def main():
    print("=" * 50)
    print("  CCTV Known Faces Bulk Insert Tool")
    print("=" * 50)

    added = 0
    failed = 0

    # ── Manual list ───────────────────────────────────────────────────────────
    for name, emp_id, dept, img_path in PERSONS:
        print(f"\n→ Adding: {name} ({emp_id})")
        if not os.path.exists(img_path):
            print(f"  ✗ Image not found: {img_path}")
            failed += 1
            continue
        result = add_person(name, emp_id, dept, img_path)
        if result.get("status") == "success":
            print(f"  ✓ ID={result['person_id']} | Embedding: {result.get('embedding_generated')}")
            added += 1
        else:
            print(f"  ✗ Error: {result.get('detail') or result}")
            failed += 1

    # ── Folder scan ───────────────────────────────────────────────────────────
    input_path = Path(INPUT_FOLDER)
    if input_path.exists():
        for person_dir in sorted(input_path.iterdir()):
            if not person_dir.is_dir():
                continue

            # Parse "John_Doe" or "John Doe" → name
            name = person_dir.name.replace("_", " ")
            images = sorted([
                f for f in person_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])

            if not images:
                print(f"\n→ Skipping '{name}' — no images found")
                continue

            print(f"\n→ Adding: {name} ({len(images)} image(s))")

            # Register with first image
            result = add_person(name, image_path=str(images[0]))
            if result.get("status") == "success":
                person_id = result["person_id"]
                print(f"  ✓ Registered (ID={person_id}) | "
                      f"Embedding: {result.get('embedding_generated')}")
                added += 1

                # Add remaining images
                for img in images[1:]:
                    img_result = add_extra_image(person_id, str(img))
                    ok = "error" not in img_result
                    print(f"  {'✓' if ok else '✗'} Extra image: {img.name}")
            else:
                print(f"  ✗ Error: {result.get('detail') or result}")
                failed += 1

    print("\n" + "=" * 50)
    print(f"  Done! Added: {added} | Failed: {failed}")
    print("=" * 50)

    if added == 0 and failed == 0:
        print("\n⚠ Nothing to add. Either:")
        print("  1. Edit the PERSONS list in this script, OR")
        print(f"  2. Create the folder '{INPUT_FOLDER}' with sub-folders per person")
        print("\nExample folder structure:")
        print("  known_faces_input/")
        print("    John_Doe/")
        print("      photo.jpg")
        print("    Jane_Smith/")
        print("      photo.jpg")


if __name__ == "__main__":
    main()
