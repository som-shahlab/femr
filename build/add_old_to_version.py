import toml

with open("pyproject.toml") as f:
    data = toml.load(f)

data["project"]["name"] += "-oldcpu"

with open("pyproject.toml", "w") as f:
    toml.dump(data, f)
