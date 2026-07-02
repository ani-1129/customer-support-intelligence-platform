from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from src.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# Standard security definitions
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Simulated in-memory database of API keys
VALID_API_KEYS = {
    "dev-api-key-123456": "Developer-Key",
    "prod-api-key-999000": "Production-Key"
}

# Simplified sliding window rate limiter in memory
# Key: client identifier (e.g. api_key or token username), Value: List of timestamps of requests
rate_limit_store: Dict[str, list] = {}

def check_rate_limit(identifier: str, limit: int = 15, window_seconds: int = 60):
    """
    Limits the requests for an identifier to `limit` per `window_seconds`.
    Throws 429 Too Many Requests if limit exceeded.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=window_seconds)
    
    # Initialize list if identifier is new
    if identifier not in rate_limit_store:
        rate_limit_store[identifier] = []
        
    # Remove expired timestamps
    rate_limit_store[identifier] = [t for t in rate_limit_store[identifier] if t > cutoff]
    
    if len(rate_limit_store[identifier]) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting more requests."
        )
        
    rate_limit_store[identifier].append(now)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Dict[str, Any]:
    """
    Authenticates requests via either OAuth2 Bearer Token or API Key.
    Returns user payload dictionary containing 'username' and 'role'.
    """
    # 1. API Key Auth
    if api_key:
        if api_key in VALID_API_KEYS:
            user = {"username": VALID_API_KEYS[api_key], "role": "admin"}
            check_rate_limit(api_key, limit=20)
            return user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )

    # 2. OAuth2 Token Auth
    if token:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role", "agent")
            if username is None:
                raise credentials_exception
            user = {"username": username, "role": role}
            check_rate_limit(username, limit=10) # Agents have lower rate limit than Admin Keys
            return user
        except JWTError:
            raise credentials_exception

    # 3. Deny if neither provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide Bearer Token or X-API-KEY header."
    )


def require_role(allowed_roles: list):
    """
    Role verification decorator dependency.
    """
    def dependency(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for your security role."
            )
        return current_user
    return dependency
