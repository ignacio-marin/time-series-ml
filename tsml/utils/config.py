import json

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path) as jsonfile:
            kwargs = json.load(jsonfile)
        return cls(**kwargs)