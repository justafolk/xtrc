# xtrc Neovim Plugin

## Install

Add `plugins/nvim` to your runtime path with your plugin manager, or source manually.

## Configuration

```lua
require("xtrc").setup({
  server_url = "http://127.0.0.1:8765",
  top_k = 8,
  repo_path = nil,
  debounce_ms = 50,
  use_telescope = true,
})
```

## Command

- `:Xtrc` opens a live Telescope picker with predictive results while typing.
- `:Xtrc <query>` opens the picker with `<query>` pre-filled.

The live query requests are debounced by `debounce_ms` (default: `50`).
