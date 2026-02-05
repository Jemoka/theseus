# Authentication Setup

The Theseus web UI uses **session-based authentication** with a login page and bcrypt password hashing for secure access.

## Quick Setup

1. **Generate a password hash:**
   ```bash
   python -m theseus.web.generate_password_hash
   ```
   Enter your desired password when prompted.

2. **Set environment variables:**
   ```bash
   export THESEUS_WEB_USERNAME='admin'
   export THESEUS_WEB_PASSWORD_HASH='$2b$12$...'  # Copy from step 1
   ```

   Or add to `.env` file:
   ```env
   THESEUS_WEB_USERNAME=admin
   THESEUS_WEB_PASSWORD_HASH=$2b$12$...
   ```

3. **Set session secret (production only):**
   ```bash
   export THESEUS_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
   ```
   (Development: auto-generated if not set, but sessions reset on restart)

4. **Start the server:**
   ```bash
   uvicorn theseus.web.app:app --host 0.0.0.0 --port 8000
   ```

5. **Access the UI:**
   Navigate to `http://localhost:8000` - you'll be redirected to the login page. Enter your credentials.

## Development Shortcut

For local development only (NOT production):
```bash
export THESEUS_WEB_USERNAME='admin'
export THESEUS_WEB_PASSWORD='mysecretpassword'  # Plain text - auto-hashed
```

This automatically hashes the password on startup.

## Security Features

✓ **Bcrypt password hashing** - Industry standard, resistant to rainbow tables
✓ **Constant-time comparison** - Prevents timing attacks on usernames
✓ **Session-based auth** - Secure cookies with HMAC signing
✓ **Login page** - Professional UI matching the dashboard design
✓ **Auto-redirect** - Unauthenticated users redirected to login
✓ **Secure by default** - If no credentials set, shows warning but allows access (for initial setup)

## Production Deployment

### Always use HTTPS in production!

Use a reverse proxy (nginx, Caddy, Traefik) to handle HTTPS:

**nginx example:**
```nginx
server {
    listen 443 ssl http2;
    server_name theseus.yourdomain.com;

    ssl_certificate /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Caddy example (automatic HTTPS):**
```
theseus.yourdomain.com {
    reverse_proxy localhost:8000
}
```

## Changing Password

1. Generate new hash:
   ```bash
   python -m theseus.web.generate_password_hash
   ```

2. Update environment variable:
   ```bash
   export THESEUS_WEB_PASSWORD_HASH='$2b$12$NEW_HASH...'
   ```

3. Restart the server

## Disabling Authentication

If you need to temporarily disable authentication (e.g., private network):
```bash
unset THESEUS_WEB_USERNAME
unset THESEUS_WEB_PASSWORD_HASH
```

The server will start without authentication and show a warning.

## Logging Out

Click the "Logout" link in the navigation (top right), or navigate to `/logout`.

## Troubleshooting

**"Incorrect username or password"**
- Double-check the password hash was copied correctly (including the `$2b$` prefix)
- Try generating a new hash
- Verify environment variables are set: `echo $THESEUS_WEB_USERNAME`

**Sessions not persisting**
- Make sure `THESEUS_SECRET_KEY` is set (production)
- Check that cookies are enabled in your browser

**Getting logged out on server restart (development)**
- This is normal if `THESEUS_SECRET_KEY` is not set
- The key is auto-generated on startup, so restart = new key = invalid sessions
- Set `THESEUS_SECRET_KEY` to persist sessions across restarts

## API Access

For programmatic API access, you'll need to maintain a session:

**Python requests (with session):**
```python
import requests

# Create session and login
session = requests.Session()
session.post(
    'http://localhost:8000/login',
    data={'username': 'admin', 'password': 'yourpassword'}
)

# Now use the session for API calls
response = session.get('http://localhost:8000/api/jobs')
print(response.json())
```

**Alternative: Use API tokens (TODO)**
For stateless API access, you'd need to implement token-based auth (JWT, API keys, etc.).
The current implementation uses sessions which require maintaining cookies.
