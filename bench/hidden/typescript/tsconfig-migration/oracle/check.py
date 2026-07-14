import json
from pathlib import Path

config = json.loads(Path("tsconfig.json").read_text())
options = config["compilerOptions"]
expected = {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": True,
    "noUncheckedIndexedAccess": True,
    "rootDir": "src",
    "outDir": "dist",
}
for key, value in expected.items():
    assert options.get(key) == value, (key, options.get(key), value)
assert Path("src/index.ts").read_text() == 'export const version: string = "1";\n'
