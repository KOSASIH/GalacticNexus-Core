from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class GalacticNexusModel(Base):
    __tablename__ = 'galactic_nexus'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(String)
