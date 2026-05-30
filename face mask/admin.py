"""
Admin CLI — System administration for the Face Recognition System.
Usage:
    python admin.py --encode            Encode faces from known_faces/
    python admin.py --unlock            Unlock the system
    python admin.py --set-password      Set/change admin password
    python admin.py --attendance        View today's attendance
    python admin.py --attendance 2026-04-06  View specific date
    python admin.py --review            Review adaptive learning queue
    python admin.py --verify-logs       Verify tamper-proof log integrity
    python admin.py --status            Show system status
"""

import argparse
import sys


def cmd_encode():
    """Encode faces from known_faces/ directory."""
    from face_encoder import encode_known_faces, load_encodings
    print("\n" + "=" * 60)
    print("  ENCODING FACES")
    print("=" * 60)
    before_total = len(load_encodings()["encodings"])
    result = encode_known_faces(delete_originals=True, preserve_existing=True)
    added = max(0, len(result["encodings"]) - before_total)
    if added:
        print(f"\n✓ Added {added} new embeddings")
        print(f"✓ Total stored: {len(result['encodings'])} embeddings for {len(set(result['names']))} people")
    else:
        print("\n✗ No faces encoded. Place images in known_faces/<Name>/")


def cmd_unlock():
    """Unlock the system with admin password."""
    from security import SecurityManager
    sm = SecurityManager()

    if not sm.is_locked():
        print("System is not locked.")
        return

    print("\n⚠ SYSTEM IS LOCKED")
    password = input("Enter admin password: ")

    if sm.admin_unlock(password):
        print("✓ System unlocked successfully.")
    else:
        print("✗ Invalid password. System remains locked.")


def cmd_set_password():
    """Set or change admin password."""
    from security import SecurityManager
    sm = SecurityManager()

    print("\n--- Set Admin Password ---")
    new_password = input("Enter new password: ")
    confirm = input("Confirm new password: ")

    if new_password != confirm:
        print("✗ Passwords do not match.")
        return

    if len(new_password) < 4:
        print("✗ Password must be at least 4 characters.")
        return

    sm.set_admin_password(new_password)
    print("✓ Admin password updated.")


def cmd_attendance(date=None):
    """View attendance records."""
    from database import AttendanceDB
    db = AttendanceDB()

    if date:
        records = db.get_attendance(date)
        print(f"\n--- Attendance for {date} ---")
    else:
        records = db.get_attendance()
        from datetime import datetime
        date = datetime.now().strftime("%Y-%m-%d")
        print(f"\n--- Attendance for Today ({date}) ---")

    if not records:
        print("No records found.")
        return

    print(f"{'Name':<20} {'Time':<10} {'Confidence':<12}")
    print("-" * 42)
    for r in records:
        conf = f"{r['confidence'] * 100:.1f}%" if r['confidence'] else "N/A"
        print(f"{r['name']:<20} {r['time']:<10} {conf:<12}")

    print(f"\nTotal: {len(records)} entries")


def cmd_review():
    """Review adaptive learning queue."""
    from adaptive_learning import AdaptiveLearner
    learner = AdaptiveLearner()

    pending = learner.get_pending_reviews()

    if not pending:
        print("\nNo pending reviews.")
        return

    print(f"\n--- Adaptive Learning Review ({len(pending)} pending) ---\n")

    for i, entry in enumerate(pending):
        print(f"  [{i}] {entry['name']} — Confidence: {entry['confidence']:.1%}")
        print(f"      Timestamp: {entry['timestamp']}")
        if entry.get('image_path'):
            print(f"      Image: {entry['image_path']}")
        print()

    while True:
        action = input("Enter index to approve (a<idx>), reject (r<idx>), or 'q' to quit: ").strip()

        if action.lower() == 'q':
            break
        elif action.startswith('a') and action[1:].isdigit():
            idx = int(action[1:])
            learner.approve_entry(idx)
            print(f"  ✓ Entry {idx} approved — embedding added")
        elif action.startswith('r') and action[1:].isdigit():
            idx = int(action[1:])
            learner.reject_entry(idx)
            print(f"  ✗ Entry {idx} rejected")
        else:
            print("  Invalid input. Use a0, r0, a1, r1, etc.")


def cmd_verify_logs():
    """Verify tamper-proof log integrity."""
    from tamper_proof_log import TamperProofLog
    log = TamperProofLog()

    print("\n--- Log Integrity Verification ---")
    result = log.verify_integrity()

    if result["valid"]:
        print(f"✓ {result['details']}")
    else:
        print(f"✗ INTEGRITY VIOLATION: {result['details']}")
        print(f"  Total entries checked: {result['total_entries']}")
        print(f"  First broken at entry: {result['first_broken_at']}")


def cmd_status():
    """Show system status overview."""
    from security import SecurityManager
    from database import AttendanceDB
    from face_encoder import load_encodings, get_registered_names
    from tamper_proof_log import TamperProofLog
    from adaptive_learning import AdaptiveLearner

    print("\n" + "=" * 60)
    print("  SYSTEM STATUS")
    print("=" * 60)

    # Security
    sm = SecurityManager()
    status = sm.get_status()
    lock_str = "🔒 LOCKED" if status["locked"] else "🔓 Unlocked"
    print(f"\n  Security:     {lock_str}")
    print(f"  Failed Tries: {status['failed_attempts']}/{status['max_attempts']}")

    # Encodings
    names = get_registered_names()
    data = load_encodings()
    print(f"\n  Registered:   {len(names)} people")
    print(f"  Embeddings:   {len(data['encodings'])} total")
    if names:
        print(f"  Names:        {', '.join(sorted(names))}")

    # Attendance
    db = AttendanceDB()
    today_count = db.get_count()
    print(f"\n  Today's Att:  {today_count} entries")

    # Log integrity
    log = TamperProofLog()
    integrity = log.verify_integrity()
    int_str = "✓ Intact" if integrity["valid"] else "✗ TAMPERED"
    print(f"\n  Log Status:   {int_str} ({integrity['total_entries']} entries)")

    # Adaptive learning
    learner = AdaptiveLearner()
    stats = learner.get_stats()
    print(f"\n  Review Queue: {stats['pending']} pending, {stats['reviewed']} reviewed")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition System — Admin CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin.py --encode             Encode faces from known_faces/
  python admin.py --unlock             Unlock locked system
  python admin.py --set-password       Change admin password
  python admin.py --attendance         Today's attendance
  python admin.py --attendance 2026-04-06   Specific date
  python admin.py --review             Review adaptive learning queue
  python admin.py --verify-logs        Check log integrity
  python admin.py --status             System overview
        """
    )

    parser.add_argument("--encode", action="store_true", help="Encode faces from known_faces/")
    parser.add_argument("--unlock", action="store_true", help="Unlock the system")
    parser.add_argument("--set-password", action="store_true", help="Set admin password")
    parser.add_argument("--attendance", nargs="?", const="today", help="View attendance")
    parser.add_argument("--review", action="store_true", help="Review adaptive learning queue")
    parser.add_argument("--verify-logs", action="store_true", help="Verify log integrity")
    parser.add_argument("--status", action="store_true", help="Show system status")

    args = parser.parse_args()

    if args.encode:
        cmd_encode()
    elif args.unlock:
        cmd_unlock()
    elif args.set_password:
        cmd_set_password()
    elif args.attendance is not None:
        date = None if args.attendance == "today" else args.attendance
        cmd_attendance(date)
    elif args.review:
        cmd_review()
    elif args.verify_logs:
        cmd_verify_logs()
    elif args.status:
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
