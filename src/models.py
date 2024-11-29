from sqlalchemy import Column, Integer, String, JSON, Enum, Time, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
from datetime import datetime

Base = declarative_base()

class EventType(enum.Enum):
    SINGLE_KNOWN = "SINGLE_KNOWN"
    FAMILY_PROFILE = "FAMILY_PROFILE"
    UNKNOWN_WITH_KNOWN = "UNKNOWN_WITH_KNOWN"
    SUSPECT = "SUSPECT"
    NO_DETECTION = "NO_DETECTION"

class EventLog(Base):
    __tablename__ = 'event_log'
    id = Column(Integer, primary_key=True)
    event_type = Column(Enum(EventType), nullable=False)
    payload = Column(JSON, nullable=False)

class FamilyMember(Base):
    __tablename__ = 'family_member'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class SystemState(Base):
    __tablename__ = 'system_state'
    id = Column(Integer, primary_key=True)
    no_detection_count = Column(Integer, default=0, nullable=False)
    no_detection_flag = Column(Integer, default=0, nullable=False)

class MusicSchedule(Base):
    __tablename__ = 'music_schedule'
    id = Column(Integer, primary_key=True)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    playlist_id = Column(String, nullable=False)

# New models for Project 2
class Faces(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    last_updated = Column(String, nullable=False, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class KnownPersons(Base):
    __tablename__ = 'known_persons'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)
    detection_count = Column(Integer, default=1, nullable=False)
    reference_image = Column(String, nullable=True)

class UnknownVisitors(Base):
    __tablename__ = 'unknown_visitors'
    ulid = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)
    visit_count = Column(Integer, default=1, nullable=False)
    camera_id = Column(String, nullable=True)
    image_path = Column(String, nullable=True)

class FamilyProfiles(Base):
    __tablename__ = 'family_profiles'
    profile_id = Column(Integer, primary_key=True)
    profile_name = Column(String, nullable=False)
    member_ids = Column(String, nullable=False)  # Comma-separated IDs
    created_at = Column(String, nullable=False)
    last_updated = Column(String, nullable=False)

class Detections(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    timestamp = Column(String, nullable=False)
    device = Column(String, nullable=False)
    status = Column(String, nullable=False)
    image_path = Column(String, nullable=True)
    camera_id = Column(String, nullable=True)
    unknown_id = Column(String, nullable=True)