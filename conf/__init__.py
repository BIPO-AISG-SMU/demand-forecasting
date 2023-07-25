import yaml
import pathlib

path = pathlib.Path(__file__).parent / "base" / "data_conf.yml"
with path.open(mode="rb") as yamlfile:
    bipo_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)
