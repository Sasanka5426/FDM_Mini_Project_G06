from pydantic import BaseModel

class AppName(BaseModel):
    # island: str
    # culmenLength: float
    # culmenDepth: float
    # flipperLength: float
    # bodyMass: float
    # sex: str
    # species: str
    name : str

    def __getitem__(self, item):
        return getattr(self, item)