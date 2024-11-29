from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import EventLog, FamilyMember, EventType
from tasks import process_event_task
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class SingleKnownPayload(BaseModel):
    timestamp: str
    device: str
    name: str
    image_base64: str

class FamilyProfilePayload(BaseModel):
    timestamp: str
    device: str
    names: conlist(str, min_length=2)
    image_base64: str

class UnknownWithKnownPayload(BaseModel):
    timestamp: str
    device: str
    name: str
    unknown_id: str
    image_base64: str

class SuspectPayload(BaseModel):
    timestamp: str
    device: str
    unknown_id: str
    image_base64: str

class NoDetectionPayload(BaseModel):
    timestamp: str
    device: str
    status: str = "no_detection"

class EventResponse(BaseModel):
    status: str
    message: str
    event_id: int

class HealthResponse(BaseModel):
    status: str
    database: str
    family_members: int | None = None

# Validate family member
def validate_family_member(name: str, db):
    return db.query(FamilyMember).filter(FamilyMember.name == name).first() is not None

# API endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    try:
        with SessionLocal() as db:
            family_members = db.query(FamilyMember).count()
        return {"status": "healthy", "database": "connected", "family_members": family_members}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "database": "disconnected"}

@app.post("/webhook/single_known", response_model=EventResponse)
async def single_known(payload: SingleKnownPayload):
    with SessionLocal() as db:
        if not validate_family_member(payload.name, db):
            raise HTTPException(status_code=400, detail=f"Family member {payload.name} not found")
        try:
            event = EventLog(
                event_type=EventType.SINGLE_KNOWN,
                payload=payload.model_dump()
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            process_event_task.delay(event.id)
            return {"status": "success", "message": f"Processed SINGLE_KNOWN for {payload.name}", "event_id": event.id}
        except Exception as e:
            logger.error(f"Error processing SINGLE_KNOWN: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/family_profile", response_model=EventResponse)
async def family_profile(payload: FamilyProfilePayload):
    with SessionLocal() as db:
        for name in payload.names:
            if not validate_family_member(name, db):
                raise HTTPException(status_code=400, detail=f"Family member {name} not found")
        try:
            event = EventLog(
                event_type=EventType.FAMILY_PROFILE,
                payload=payload.model_dump()
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            process_event_task.delay(event.id)
            return {"status": "success", "message": f"Processed FAMILY_PROFILE for {', '.join(payload.names)}", "event_id": event.id}
        except Exception as e:
            logger.error(f"Error processing FAMILY_PROFILE: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/unknown_with_known", response_model=EventResponse)
async def unknown_with_known(payload: UnknownWithKnownPayload):
    with SessionLocal() as db:
        if not validate_family_member(payload.name, db):
            raise HTTPException(status_code=400, detail=f"Family member {payload.name} not found")
        try:
            event = EventLog(
                event_type=EventType.UNKNOWN_WITH_KNOWN,
                payload=payload.model_dump()
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            process_event_task.delay(event.id)
            return {"status": "success", "message": f"Processed UNKNOWN_WITH_KNOWN for {payload.name} with guest", "event_id": event.id}
        except Exception as e:
            logger.error(f"Error processing UNKNOWN_WITH_KNOWN: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/suspect", response_model=EventResponse)
async def suspect(payload: SuspectPayload):
    with SessionLocal() as db:
        try:
            event = EventLog(
                event_type=EventType.SUSPECT,
                payload=payload.model_dump()
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            process_event_task.delay(event.id)
            return {"status": "success", "message": "Processed SUSPECT", "event_id": event.id}
        except Exception as e:
            logger.error(f"Error processing SUSPECT: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/no_detection", response_model=EventResponse)
async def no_detection(payload: NoDetectionPayload):
    with SessionLocal() as db:
        try:
            event = EventLog(
                event_type=EventType.NO_DETECTION,
                payload=payload.model_dump()
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            process_event_task.delay(event.id)
            return {"status": "success", "message": "Processed NO_DETECTION", "event_id": event.id}
        except Exception as e:
            logger.error(f"Error processing NO_DETECTION: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))