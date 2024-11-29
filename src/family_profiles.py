import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import FamilyProfiles
from dotenv import load_dotenv
import os
from threading import Lock

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    filename='/app/logs/family_profiles.log'
)

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database lock for thread safety
family_profiles_db_lock = Lock()

def init_family_profiles_db():
    """Initialize the family profiles database (handled by init_db.sql)."""
    with family_profiles_db_lock:
        logger.debug("Ensuring family profiles table exists via SQLAlchemy")
        # Table is created via init_db.sql or SQLAlchemy Base.metadata.create_all
        logger.info("Family profiles database ready")

def check_family_profile(member_ids):
    """Check if a family profile exists for the given member IDs."""
    with family_profiles_db_lock:
        logger.debug(f"Checking family profile for member IDs: {member_ids}")
        with SessionLocal() as db:
            member_ids_str = ','.join(sorted(map(str, member_ids)))
            profile = db.query(FamilyProfiles).filter(FamilyProfiles.member_ids == member_ids_str).first()
            if profile:
                logger.info(f"Found family profile: {profile.profile_name} for member IDs: {member_ids_str}")
                return profile.profile_name
            logger.debug(f"No family profile found for member IDs: {member_ids_str}")
            return None

def add_family_profile(profile_name, member_ids, timestamp):
    """Add a new family profile to the database."""
    with family_profiles_db_lock:
        logger.debug(f"Adding family profile: {profile_name} with member IDs: {member_ids}")
        with SessionLocal() as db:
            member_ids_str = ','.join(sorted(map(str, member_ids)))
            profile = FamilyProfiles(
                profile_name=profile_name,
                member_ids=member_ids_str,
                created_at=timestamp,
                last_updated=timestamp
            )
            db.add(profile)
            db.commit()
            logger.info(f"Added family profile: {profile_name} with member IDs: {member_ids_str}")