# Native turn policy

The experimental WASM decision engine and shadow NativeDirector were removed.
The interactive agent has one native authority for model requests, recovery,
verification, and settlement. Remove `HI_ENGINE_*` overrides and `/engine`
commands. Historical saved settings remain readable; selecting a removed engine
produces a migration error. The managed RSI trust domain remains separate.
