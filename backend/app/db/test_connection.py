import psycopg2
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_db_connection():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=settings.POSTGRES_SERVER,
            database=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD
        )
        
        # Open a cursor
        cur = conn.cursor()
        
        # Test PostGIS extension
        cur.execute("SELECT PostGIS_version();")
        postgis_version = cur.fetchone()[0]
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        logger.info(f"Successfully connected to PostgreSQL database")
        logger.info(f"PostGIS version: {postgis_version}")
        return True
    
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False

if __name__ == "__main__":
    test_db_connection()