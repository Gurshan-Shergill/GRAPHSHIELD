from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

from app.core.config import get_settings

settings = get_settings()

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
http_bearer = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    to_encode.update({'exp': expire, 'type': 'access'})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    to_encode.update({'exp': expire, 'type': 'refresh'})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )


def generate_api_key() -> tuple[str, str]:
    '''Returns (api_key, key_hash)'''
    raw_key = f'{settings.api_key_prefix}{secrets.token_urlsafe(32)}'
    key_hash = get_password_hash(raw_key)
    return raw_key, key_hash


def verify_api_key(raw_key: str, key_hash: str) -> bool:
    return pwd_context.verify(raw_key, key_hash)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> int:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Not authenticated',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    payload = decode_token(credentials.credentials)
    if payload.get('type') != 'access':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid token type',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    user_id = payload.get('sub')
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid token',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return int(user_id)


async def get_current_user_org_id(
    request: Request,
    user_id: int = Depends(get_current_user_id),
) -> int:
    # In real implementation, fetch user from DB and return org_id
    # For now, return from request state if set by middleware
    return getattr(request.state, 'org_id', 1)


async def verify_api_key_dependency(
    request: Request,
    api_key: str = Depends(lambda r: r.headers.get('X-API-Key')),
) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='API key required',
            headers={'WWW-Authenticate': 'ApiKey'},
        )
    # In real implementation, verify against hashed keys in DB
    # For now, accept any key starting with prefix
    if not api_key.startswith(settings.api_key_prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid API key format',
            headers={'WWW-Authenticate': 'ApiKey'},
        )
    return api_key
