# Complete FastAPI Guide: From Basics to Advanced

## Table of Contents
1. [What is FastAPI?](#1-what-is-fastapi)
2. [Why FastAPI?](#2-why-fastapi)
3. [FastAPI Fundamentals](#3-fastapi-fundamentals)
4. [Request and Response Models](#4-request-and-response-models)
5. [Routing and Endpoints](#5-routing-and-endpoints)
6. [Path Parameters and Query Parameters](#6-path-parameters-and-query-parameters)
7. [Request Body and Validation](#7-request-body-and-validation)
8. [Dependencies and Dependency Injection](#8-dependencies-and-dependency-injection)
9. [Error Handling](#9-error-handling)
10. [Middleware](#10-middleware)
11. [Authentication and Security](#11-authentication-and-security)
12. [Database Integration](#12-database-integration)
13. [Background Tasks](#13-background-tasks)
14. [WebSockets](#14-websockets)
15. [Testing FastAPI](#15-testing-fastapi)
16. [Deployment](#16-deployment)
17. [Best Practices](#17-best-practices)
18. [Our RAG System Implementation](#18-our-rag-system-implementation)

---

## 1. What is FastAPI?

### 1.1 Definition

**FastAPI** is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

### 1.2 Key Features

- ⚡ **Fast**: One of the fastest Python frameworks (comparable to NodeJS and Go)
- 📝 **Easy**: Designed to be easy to use and learn
- 🔒 **Standards-based**: Based on (and fully compatible with) open standards:
  - OpenAPI (formerly Swagger)
  - JSON Schema
  - OAuth2
- 🎯 **Type Hints**: Built on Python type hints for better IDE support and validation
- 📚 **Auto Documentation**: Automatic interactive API documentation
- 🔄 **Async Support**: Native support for async/await

### 1.3 FastAPI vs Other Frameworks

| Framework | Speed | Learning Curve | Async | Auto Docs |
|-----------|-------|----------------|-------|-----------|
| **FastAPI** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |
| Flask | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| Django | ⭐⭐ | ⭐⭐⭐ | Partial | ❌ |
| Express.js | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ |

---

## 2. Why FastAPI?

### 2.1 Performance

**Benchmarks** (requests per second):
- FastAPI: ~50,000 req/s
- Flask: ~20,000 req/s
- Django: ~10,000 req/s

**Why FastAPI is Fast**:
- Built on Starlette (async framework)
- Uses Pydantic for validation (compiled with Rust)
- No overhead from unnecessary features

### 2.2 Developer Experience

**Automatic Validation**:
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# FastAPI automatically validates request body
@app.post("/users")
async def create_user(user: User):
    # user is already validated!
    return user
```

**Auto Documentation**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

**Type Safety**:
```python
# IDE knows the types!
@app.get("/users/{user_id}")
async def get_user(user_id: int) -> User:
    # IDE autocomplete works!
    return User(name="John", age=30)
```

### 2.3 Modern Python Features

- Async/await support
- Type hints throughout
- Dependency injection
- WebSocket support
- Background tasks

---

## 3. FastAPI Fundamentals

### 3.1 Installation

```bash
pip install fastapi uvicorn[standard]
```

### 3.2 Basic Application

**Minimal Example**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

**Run the server**:
```bash
uvicorn main:app --reload
```

### 3.3 Application Instance

**Creating the App**:
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="API description",
    version="1.0.0",
    docs_url="/docs",  # Custom docs URL
    redoc_url="/redoc"  # Custom ReDoc URL
)
```

**Common Parameters**:
- `title`: API title
- `description`: API description
- `version`: API version
- `docs_url`: Swagger UI URL (set to None to disable)
- `redoc_url`: ReDoc URL
- `openapi_url`: OpenAPI schema URL

### 3.4 Route Decorators

**HTTP Methods**:
```python
@app.get("/")      # GET request
@app.post("/")     # POST request
@app.put("/")      # PUT request
@app.delete("/")   # DELETE request
@app.patch("/")    # PATCH request
@app.options("/")  # OPTIONS request
@app.head("/")     # HEAD request
```

**Our Example** (`src/api.py`):
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ingest")
async def ingest_document():
    # Process document
    pass

@app.post("/query")
async def query_document(request: QueryRequest):
    # Query document
    pass
```

---

## 4. Request and Response Models

### 4.1 Pydantic Models

**What is Pydantic?**
- Data validation using Python type hints
- Automatic JSON serialization/deserialization
- Editor support and autocomplete

**Basic Model**:
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str
```

### 4.2 Request Models

**Our Example** (`src/api.py`):
```python
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
```

**Advanced Models**:
```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=120)
    email: EmailStr
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com"
            }
        }
```

### 4.3 Response Models

**Basic Response**:
```python
@app.post("/users")
async def create_user(user: User) -> User:
    return user  # Automatically serialized to JSON
```

**Response Model**:
```python
class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.post("/users", response_model=UserResponse)
async def create_user(user: User):
    # Return UserResponse, not User
    return UserResponse(id=1, name=user.name, email=user.email)
```

**Our Example**:
```python
class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    return QueryResponse(answer="...", sources=[...])
```

### 4.4 Field Validation

**Field Constraints**:
```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)  # Greater than 0
    quantity: int = Field(..., ge=0)  # Greater than or equal to 0
    description: Optional[str] = Field(None, max_length=500)
```

**Custom Validators**:
```python
from pydantic import BaseModel, validator

class User(BaseModel):
    email: str
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v
```

---

## 5. Routing and Endpoints

### 5.1 Basic Routes

**Simple Route**:
```python
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

**Route with Path**:
```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
```

### 5.2 Route Order Matters

**Important**: More specific routes should come first!

```python
# ✅ Correct order
@app.get("/users/me")  # Specific first
async def get_current_user():
    return {"user": "current"}

@app.get("/users/{user_id}")  # Generic second
async def get_user(user_id: int):
    return {"user_id": user_id}
```

### 5.3 Route Prefixes

**Using APIRouter**:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

@router.get("/users")
async def get_users():
    return {"users": []}

# Include in main app
app.include_router(router)
```

**Our Project Structure** (if we refactored):
```python
# src/routers/query.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query")
async def query_document(request: QueryRequest):
    # Query logic
    pass

# src/api.py
from src.routers import query
app.include_router(query.router)
```

### 5.4 Tags and Summary

**Organizing Endpoints**:
```python
@app.post(
    "/query",
    tags=["Query"],
    summary="Query the document",
    description="Query the RAG system with a question",
    response_description="Answer with sources"
)
async def query_document(request: QueryRequest):
    """Query the document using RAG."""
    pass
```

---

## 6. Path Parameters and Query Parameters

### 6.1 Path Parameters

**Basic Path Parameter**:
```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

**Path Parameter with Type**:
```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):  # FastAPI validates it's an int
    return {"user_id": user_id}
```

**Multiple Path Parameters**:
```python
@app.get("/users/{user_id}/items/{item_id}")
async def get_user_item(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}
```

**Path Parameter with Enum**:
```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

### 6.2 Query Parameters

**Basic Query Parameter**:
```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

**Optional Query Parameters**:
```python
from typing import Optional

@app.get("/items/")
async def read_items(
    q: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    return {"q": q, "skip": skip, "limit": limit}
```

**Query Parameter Validation**:
```python
from fastapi import Query

@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(None, min_length=3, max_length=50),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    return {"q": q, "skip": skip, "limit": limit}
```

**Query Parameter with Description**:
```python
@app.get("/items/")
async def read_items(
    q: Optional[str] = Query(
        None,
        description="Search query",
        example="python"
    )
):
    return {"q": q}
```

### 6.3 Combining Path and Query Parameters

```python
@app.get("/users/{user_id}/items")
async def get_user_items(
    user_id: int,  # Path parameter
    skip: int = 0,  # Query parameter
    limit: int = 10  # Query parameter
):
    return {
        "user_id": user_id,
        "items": [],
        "skip": skip,
        "limit": limit
    }
```

---

## 7. Request Body and Validation

### 7.1 Simple Request Body

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

**Our Example**:
```python
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_document(request: QueryRequest):
    # request.question is automatically validated
    return process_query(request.question)
```

### 7.2 Multiple Request Bodies

**Multiple Parameters**:
```python
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,  # Path parameter
    item: Item,    # Request body
    user: User     # Another request body
):
    return {"item_id": item_id, "item": item, "user": user}
```

**Body + Query Parameters**:
```python
@app.post("/items/")
async def create_item(
    item: Item,           # Request body
    importance: int = 1   # Query parameter
):
    return {"item": item, "importance": importance}
```

### 7.3 Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class User(BaseModel):
    name: str
    email: str
    address: Address  # Nested model

@app.post("/users/")
async def create_user(user: User):
    return user
```

### 7.4 List of Items

```python
@app.post("/items/")
async def create_items(items: List[Item]):
    return {"items": items}
```

### 7.5 File Uploads

```python
from fastapi import UploadFile, File

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "size": len(contents)
    }
```

**Multiple Files**:
```python
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [f.filename for f in files]}
```

---

## 8. Dependencies and Dependency Injection

### 8.1 What are Dependencies?

**Dependencies** allow you to:
- Share code between routes
- Handle authentication
- Database connections
- Configuration
- Reusable logic

### 8.2 Basic Dependency

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

### 8.3 Dependency with Parameters

```python
def get_query_params(
    skip: int = 0,
    limit: int = 10
):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(params: dict = Depends(get_query_params)):
    return params
```

### 8.4 Class-based Dependencies

```python
class QueryParams:
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(params: QueryParams = Depends()):
    return {"skip": params.skip, "limit": params.limit}
```

### 8.5 Dependency Chains

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(db: Session = Depends(get_db)):
    # Use db from dependency
    return db.query(User).first()

@app.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return user
```

### 8.6 Sub-dependencies

```python
def get_query(q: Optional[str] = None):
    return q

def get_filter(f: Optional[str] = None, q: str = Depends(get_query)):
    return {"q": q, "f": f}

@app.get("/items/")
async def read_items(filter: dict = Depends(get_filter)):
    return filter
```

### 8.7 Dependency Overrides (Testing)

```python
from fastapi.testclient import TestClient

def override_get_db():
    # Use test database
    return TestDB()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)
```

### 8.8 Our Project Example

**Current Implementation** (using global variables):
```python
# src/api.py
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None

@app.post("/query")
async def query_document(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(...)
    return rag_pipeline.query(request.question)
```

**Better with Dependencies**:
```python
def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    return rag_pipeline

@app.post("/query")
async def query_document(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    return pipeline.query(request.question)
```

---

## 9. Error Handling

### 9.1 HTTPException

**Basic Exception**:
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    return items[item_id]
```

**Our Example**:
```python
@app.post("/query")
async def query_document(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document not ingested yet. Please call /ingest first."
        )
```

### 9.2 Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": exc.message}
    )
```

### 9.3 Validation Error Handler

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )
```

### 9.4 Status Codes

**Common Status Codes**:
```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    # No content returned
    pass
```

**Status Code Constants**:
- `status.HTTP_200_OK` (200)
- `status.HTTP_201_CREATED` (201)
- `status.HTTP_400_BAD_REQUEST` (400)
- `status.HTTP_401_UNAUTHORIZED` (401)
- `status.HTTP_404_NOT_FOUND` (404)
- `status.HTTP_500_INTERNAL_SERVER_ERROR` (500)

---

## 10. Middleware

### 10.1 What is Middleware?

**Middleware** is code that runs:
- Before each request
- After each request
- Can modify requests/responses
- Can add headers, logging, etc.

### 10.2 Basic Middleware

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 10.3 CORS Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10.4 Logging Middleware

```python
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
```

### 10.5 Authentication Middleware

```python
from fastapi import Request, HTTPException

@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        token = request.headers.get("Authorization")
        if not token or not verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
    response = await call_next(request)
    return response
```

---

## 11. Authentication and Security

### 11.1 Basic Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials"
        )
    return credentials.username

@app.get("/protected")
async def protected_route(username: str = Depends(verify_credentials)):
    return {"message": f"Hello {username}"}
```

### 11.2 JWT Authentication

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

@app.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"user": current_user}
```

### 11.3 API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "secret-api-key":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

@app.get("/protected")
async def protected_route(api_key: str = Depends(verify_api_key)):
    return {"message": "Access granted"}
```

### 11.4 OAuth2 with Password

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Verify username and password
    if form_data.username != "admin" or form_data.password != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    # Create and return token
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}
```

---

## 12. Database Integration

### 12.1 SQLAlchemy Integration

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

### 12.2 Async Database (SQLAlchemy 2.0)

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@app.post("/users/")
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
```

---

## 13. Background Tasks

### 13.1 Background Tasks

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(message)

@app.post("/send-email/")
async def send_email(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Email sent to {email}")
    return {"message": "Email sent"}
```

### 13.2 Celery for Complex Tasks

```python
from celery import Celery

celery_app = Celery("tasks", broker="redis://localhost:6379")

@celery_app.task
def process_document(document_id: int):
    # Long-running task
    pass

@app.post("/process/")
async def process(document_id: int):
    process_document.delay(document_id)
    return {"message": "Processing started"}
```

---

## 14. WebSockets

### 14.1 Basic WebSocket

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message: {data}")
```

### 14.2 WebSocket with Connection Manager

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Message: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## 15. Testing FastAPI

### 15.1 TestClient

```python
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.0}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Item"
```

### 15.2 Testing with Dependencies

```python
from fastapi.testclient import TestClient

def override_get_db():
    # Use test database
    return TestDB()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)
```

### 15.3 Async Testing

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_read_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
```

### 15.4 Testing Our RAG System

```python
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_ingest_document():
    response = client.post("/ingest")
    assert response.status_code == 200
    assert "chunks_created" in response.json()

def test_query_document():
    # First ingest
    client.post("/ingest")
    
    # Then query
    response = client.post(
        "/query",
        json={"question": "What is Python?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()
```

---

## 16. Deployment

### 16.1 Running with Uvicorn

**Development**:
```bash
uvicorn main:app --reload
```

**Production**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 16.2 Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

### 16.3 Cloud Deployment

**AWS (Elastic Beanstalk)**:
```bash
eb init
eb create
eb deploy
```

**Google Cloud (Cloud Run)**:
```bash
gcloud run deploy
```

**Heroku**:
```bash
heroku create
git push heroku main
```

### 16.4 Environment Variables

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    database_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 17. Best Practices

### 17.1 Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── items.py
│   │   └── users.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── dependencies.py
├── tests/
│   └── test_main.py
├── requirements.txt
└── Dockerfile
```

### 17.2 Code Organization

**Separate Routers**:
```python
# app/routers/items.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/items/")
async def get_items():
    return []

# app/main.py
from app.routers import items
app.include_router(items.router, prefix="/api")
```

**Separate Models**:
```python
# app/models/schemas.py
from pydantic import BaseModel

class ItemBase(BaseModel):
    name: str

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    class Config:
        from_attributes = True
```

### 17.3 Error Handling

```python
# Custom exceptions
class ItemNotFoundError(Exception):
    pass

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": "Item not found"}
    )
```

### 17.4 Logging

```python
import logging

logger = logging.getLogger(__name__)

@app.post("/items/")
async def create_item(item: Item):
    logger.info(f"Creating item: {item.name}")
    try:
        # Create item
        pass
    except Exception as e:
        logger.error(f"Error creating item: {str(e)}", exc_info=True)
        raise
```

### 17.5 Response Models

**Always use response models**:
```python
@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):
    # Return Item, not ItemCreate
    return Item(id=1, name=item.name)
```

### 17.6 Validation

**Use Pydantic for all validation**:
```python
from pydantic import BaseModel, Field, validator

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

---

## 18. Our RAG System Implementation

### 18.1 Current Structure

**File**: `src/api.py`

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise RAG System",
    description="RAG system for querying documents",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document():
    # Document ingestion logic
    pass

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    # Query logic
    pass
```

### 18.2 Improvements We Could Make

**1. Use Dependencies**:
```python
def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    return rag_pipeline

@app.post("/query")
async def query_document(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    return pipeline.query(request.question)
```

**2. Add Error Handling**:
```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )
```

**3. Add Logging Middleware**:
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response
```

**4. Add Rate Limiting**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query_document(request: Request, query: QueryRequest):
    # Query logic
    pass
```

**5. Add Authentication**:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=401)
    return api_key

@app.post("/query")
async def query_document(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    # Query logic
    pass
```

### 18.3 Refactored Structure

**Better Organization**:
```
src/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── dependencies.py   # Dependencies
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── query.py     # Query endpoints
│   │   └── ingest.py    # Ingest endpoints
│   └── models/
│       └── schemas.py   # Pydantic models
```

**Example Router**:
```python
# src/api/routers/query.py
from fastapi import APIRouter, Depends, HTTPException
from src.api.models.schemas import QueryRequest, QueryResponse
from src.api.dependencies import get_rag_pipeline

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    pipeline = Depends(get_rag_pipeline)
):
    return pipeline.query(request.question)
```

---

## Summary

### Key Takeaways

1. **FastAPI is Fast**: Built on Starlette with async support
2. **Type Safety**: Uses Pydantic for validation
3. **Auto Documentation**: Swagger UI and ReDoc automatically generated
4. **Dependency Injection**: Clean, reusable code
5. **Modern Python**: Async/await, type hints, modern patterns

### Quick Reference

**Basic Endpoint**:
```python
@app.get("/")
async def root():
    return {"message": "Hello"}
```

**With Path Parameter**:
```python
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}
```

**With Request Body**:
```python
@app.post("/items/")
async def create_item(item: Item):
    return item
```

**With Dependencies**:
```python
@app.get("/items/")
async def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

**With Error Handling**:
```python
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Not found")
    return items[item_id]
```

### Next Steps

1. Practice with simple endpoints
2. Add request/response models
3. Implement dependencies
4. Add error handling
5. Add authentication
6. Deploy to production

Happy FastAPI coding! 🚀
