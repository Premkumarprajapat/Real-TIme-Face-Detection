"""
Face encoder for converting stored face images into 128-d embeddings.
Supports incremental training so new users can be added without
overwriting embeddings that are already stored.
"""

import os
import pickle
import shutil
import stat

import face_recognition

import config


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _empty_encodings() -> dict:
    """Return the default payload used across the app."""
    return {"names": [], "encodings": []}


def _remove_readonly(func, path, _excinfo):
    """Retry folder deletion after clearing the read-only flag on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _load_existing_encodings() -> dict:
    """Load encodings from disk without emitting startup warnings."""
    if not os.path.exists(config.ENCODINGS_FILE):
        return _empty_encodings()

    try:
        with open(config.ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, TypeError, ValueError):
        return _empty_encodings()

    return {
        "names": list(data.get("names", [])),
        "encodings": list(data.get("encodings", [])),
    }


def _save_encodings(data: dict) -> None:
    """Persist encodings to disk."""
    os.makedirs(config.ENCODINGS_DIR, exist_ok=True)
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def encode_known_faces(delete_originals: bool = True, preserve_existing: bool = True) -> dict:
    """
    Encode any pending images stored under known_faces/<PersonName>/.

    When preserve_existing is True, new embeddings are appended to the
    existing database instead of rebuilding it from scratch.
    Only the source images for successfully processed people are deleted.
    """
    existing_data = _load_existing_encodings() if preserve_existing else _empty_encodings()
    encodings_data = {
        "names": list(existing_data["names"]),
        "encodings": list(existing_data["encodings"]),
    }

    if not os.path.exists(config.KNOWN_FACES_DIR):
        print("[ENCODER] No known_faces directory found.")
        return encodings_data

    people = [
        d
        for d in os.listdir(config.KNOWN_FACES_DIR)
        if os.path.isdir(os.path.join(config.KNOWN_FACES_DIR, d))
    ]

    if not people:
        print("[ENCODER] No person directories found in known_faces/")
        return encodings_data

    print(f"[ENCODER] Found {len(people)} people to encode...")

    processed_people = []
    new_embeddings = 0

    for person_name in people:
        person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if not image_files:
            print(f"  [SKIP] {person_name}: No images found")
            continue

        person_encodings = []
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                image = face_recognition.load_image_file(img_path)
                face_encs = face_recognition.face_encodings(
                    image,
                    num_jitters=config.FACE_ENCODING_JITTERS,
                )
                if face_encs:
                    person_encodings.append(face_encs[0])
                    print(f"  [OK] {person_name}/{img_file} encoded")
                else:
                    print(f"  [WARN] {person_name}/{img_file} no face detected")
            except Exception as e:
                print(f"  [ERROR] {person_name}/{img_file} {e}")

        if not person_encodings:
            continue

        encodings_data["names"].extend([person_name] * len(person_encodings))
        encodings_data["encodings"].extend(person_encodings)
        processed_people.append(person_name)
        new_embeddings += len(person_encodings)
        print(f"  [DONE] {person_name}: {len(person_encodings)} encodings")

    if new_embeddings:
        _save_encodings(encodings_data)
        print(
            f"[ENCODER] Added {new_embeddings} new encodings. "
            f"Total stored: {len(encodings_data['encodings'])}"
        )
    else:
        print("[ENCODER] No new encodings were created.")

    if delete_originals and processed_people:
        for person_name in processed_people:
            person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
            if not os.path.exists(person_dir):
                continue
            try:
                shutil.rmtree(person_dir, onerror=_remove_readonly)
                print(f"  [PRIVACY] Deleted original images for {person_name}")
            except Exception as e:
                print(f"  [WARN] Could not delete {person_name} folder: {e}")
                print("         You may delete it manually for privacy.")
        print("[ENCODER] Processed original images deleted for privacy.")

    return encodings_data


def load_encodings() -> dict:
    """Load saved encodings from disk."""
    if not os.path.exists(config.ENCODINGS_FILE):
        print("[ENCODER] No encodings file found. Run encoding first.")
        return _empty_encodings()

    data = _load_existing_encodings()
    print(f"[ENCODER] Loaded {len(data['encodings'])} encodings for {len(set(data['names']))} people")
    return data


def add_encoding(name: str, encoding) -> None:
    """Add one encoding to the existing set without retraining everything."""
    data = _load_existing_encodings()
    data["names"].append(name)
    data["encodings"].append(encoding)
    _save_encodings(data)
    print(f"[ENCODER] Added new encoding for {name}. Total: {len(data['encodings'])}")


def get_registered_names() -> list:
    """Get a unique list of registered names."""
    data = load_encodings()
    return list(set(data["names"]))


if __name__ == "__main__":
    print("=" * 60)
    print("  FACE ENCODER - Incremental Encoding Pipeline")
    print("=" * 60)
    print(f"\nScanning: {config.KNOWN_FACES_DIR}")
    print("Place images in: known_faces/<PersonName>/image.jpg\n")

    before = len(_load_existing_encodings()["encodings"])
    result = encode_known_faces(delete_originals=True, preserve_existing=True)
    after = len(result["encodings"])
    added = max(0, after - before)

    if added:
        print(f"\nSuccessfully added {added} new embeddings")
        print("Processed source images were deleted for privacy")
    else:
        print("\nNo faces were encoded. Check your known_faces/ directory.")
