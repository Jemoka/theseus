#!/usr/bin/env python3
"""
Generate a bcrypt password hash for use with THESEUS_WEB_PASSWORD_HASH.

Usage:
    python -m theseus.web.generate_password_hash
"""

import getpass
import sys

try:
    import bcrypt
except ImportError:
    print("Error: bcrypt not installed. Install with: pip install bcrypt")
    sys.exit(1)


def main():
    print("Theseus Web Password Hash Generator")
    print("=" * 40)
    print()

    password = getpass.getpass("Enter password: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Error: Passwords do not match!")
        sys.exit(1)

    if len(password) < 8:
        print(
            "Warning: Password is less than 8 characters. Consider using a longer password."
        )

    # Generate hash
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    print()
    print("Password hash generated successfully!")
    print()
    print("Add these to your environment:")
    print("-" * 40)
    print("export THESEUS_WEB_USERNAME='admin'")
    print(f"export THESEUS_WEB_PASSWORD_HASH='{password_hash}'")
    print("-" * 40)
    print()
    print("Or add to your .env file:")
    print("-" * 40)
    print("THESEUS_WEB_USERNAME=admin")
    print(f"THESEUS_WEB_PASSWORD_HASH={password_hash}")
    print("-" * 40)


if __name__ == "__main__":
    main()
