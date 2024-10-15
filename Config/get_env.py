from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    rootPath: str
    rtsp_url: str
    video_path: str
    source_id: int
    access_key: str
    secret_key: str
    connection_string: str
    port: int

    class Config:
        env_file = "./.env"
        extra = "allow" 

