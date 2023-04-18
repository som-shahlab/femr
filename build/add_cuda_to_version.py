import toml

with open('pyproject.toml') as f:
  data = toml.load(f)

data['project']['version'] += '+cuda'

with open('pyproject.toml', 'w') as f:
  toml.dump(data, f)

