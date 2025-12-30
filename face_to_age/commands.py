import fire

from face_to_age.infer import infer  # noqa: F401
from face_to_age.train import train  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
