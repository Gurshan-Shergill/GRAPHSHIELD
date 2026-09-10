from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = 'bearer'


class TokenPayload(BaseModel):
    sub: int
    org_id: int
    role: str
    exp: int
    type: str


class UserCreate(BaseModel):
    org_name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    org_id: int
    email: str
    role: str
    is_active: bool
    last_login: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default=['papers:read', 'papers:write', 'reports:read'])
    rate_limit: int = Field(default=100, ge=1, le=10000)
    expires_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key: str  # Only returned on creation
    scopes: List[str]
    rate_limit: int
    rate_limit_window: int
    expires_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    id: int
    name: str
    scopes: List[str]
    rate_limit: int
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
