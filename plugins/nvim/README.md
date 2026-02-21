# xtrc Neovim Plugin

## Install

Add `plugins/nvim` to your runtime path with your plugin manager, or source manually.

## Configuration

```lua
require("xtrc").setup({
  server_url = "http://127.0.0.1:8765",
  top_k = 8,
  repo_path = nil,
})
```

## Command

- `:xtrc <query>`

If `<query>` is omitted, the plugin prompts for it.
