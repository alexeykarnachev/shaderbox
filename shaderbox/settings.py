from urllib.parse import urljoin

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    modelbox_url: str = "http://0.0.0.0:8228"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def modelbox_image_to_image_url(self) -> str:
        return urljoin(self.modelbox_url, "image_to_image")


settings = Settings()
