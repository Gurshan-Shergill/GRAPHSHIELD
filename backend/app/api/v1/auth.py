from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import secrets

from app.core.config import get_settings
from app.core.security import (
    verify_password, get_password_hash, create_access_token, create_refresh_token,
    generate_api_key, verify_api_key, get_current_user_id, get_current_user_org_id,
    verify_api_key_dependency
)
from app.core.database import get_session
from app.models import User, Organization, APIKey, UserRole
from app.schemas.auth import (
    Token, UserCreate, UserLogin, UserResponse, APIKeyCreate, APIKeyResponse, APIKeyListResponse
)

settings = get_settings()
router = APIRouter(prefix='/auth', tags=['Authentication'])


@router.post('/register', response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_session)):
    '''Register new organization with admin user'''
    # Check if org slug exists
    slug = user_data.org_name.lower().replace(' ', '-').replace('_', '-')
    existing_org = await db.execute(select(Organization).where(Organization.slug == slug))
    if existing_org.scalar_one_or_none():
        # Try with suffix
        base_slug = slug
        i = 1
        while True:
            existing = await db.execute(select(Organization).where(Organization.slug == f'{base_slug}-{i}'))
            if not existing.scalar_one_or_none():
                slug = f'{base_slug}-{i}'
                break
            i += 1

    # Create organization
    org = Organization(name=user_data.org_name, slug=slug)
    db.add(org)
    await db.flush()

    # Create admin user
    hashed_pw = get_password_hash(user_data.password)
    user = User(
        org_id=org.id,
        email=user_data.email,
        hashed_password=hashed_pw,
        role=UserRole.ORG_ADMIN,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await db.refresh(org)

    return UserResponse.model_validate(user)


@router.post('/login', response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_session)):
    '''Login with email/password -> returns JWT tokens'''
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Account disabled')

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    access_token = create_access_token({'sub': str(user.id), 'org_id': user.org_id, 'role': user.role.value})
    refresh_token = create_refresh_token({'sub': str(user.id), 'org_id': user.org_id})

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post('/refresh', response_model=Token)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_session)):
    '''Refresh access token using refresh token'''
    from app.core.security import decode_token
    payload = decode_token(refresh_token)
    if payload.get('type') != 'refresh':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token type')

    user_id = int(payload.get('sub'))
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found or disabled')

    access_token = create_access_token({'sub': str(user.id), 'org_id': user.org_id, 'role': user.role.value})
    new_refresh = create_refresh_token({'sub': str(user.id), 'org_id': user.org_id})

    return Token(access_token=access_token, refresh_token=new_refresh)


@router.get('/me', response_model=UserResponse)
async def get_current_user(user_id: int = Depends(get_current_user_id), db: AsyncSession = Depends(get_session)):
    '''Get current user profile'''
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    return UserResponse.model_validate(user)


# API Key management
@router.post('/api-keys', response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: APIKeyCreate,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Create new API key for organization'''
    raw_key, key_hash = generate_api_key()
    expires_at = None
    if key_data.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_days)

    api_key = APIKey(
        org_id=org_id,
        name=key_data.name,
        key_hash=key_hash,
        scopes=key_data.scopes,
        rate_limit=key_data.rate_limit,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    # Return with raw key (only time it is shown)
    response = APIKeyResponse.model_validate(api_key)
    response.key = raw_key
    return response


@router.get('/api-keys', response_model=list[APIKeyListResponse])
async def list_api_keys(
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''List all API keys for organization (without raw keys)'''
    result = await db.execute(select(APIKey).where(APIKey.org_id == org_id).order_by(APIKey.created_at.desc()))
    keys = result.scalars().all()
    return [APIKeyListResponse.model_validate(k) for k in keys]


@router.delete('/api-keys/{key_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    org_id: int = Depends(get_current_user_org_id),
    db: AsyncSession = Depends(get_session),
):
    '''Revoke API key'''
    result = await db.execute(select(APIKey).where(APIKey.id == key_id, APIKey.org_id == org_id))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='API key not found')
    api_key.is_active = False
    await db.commit()
