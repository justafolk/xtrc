local M = {}

M.config = {
  server_url = "http://127.0.0.1:8765",
  top_k = 8,
  repo_path = nil,
}

local function notify(msg, level)
  vim.notify(msg, level or vim.log.levels.INFO, { title = "xtrc" })
end

local function resolve_repo()
  return M.config.repo_path or vim.fn.getcwd()
end

local function open_result(result, repo_path)
  local file_path = result.file_path
  if string.sub(file_path, 1, 1) ~= "/" then
    file_path = repo_path .. "/" .. file_path
  end

  vim.cmd("edit " .. vim.fn.fnameescape(file_path))
  local line = tonumber(result.start_line) or 1
  pcall(vim.api.nvim_win_set_cursor, 0, { line, 0 })
end

local function pick_and_open(results, repo_path)
  if #results == 0 then
    notify("No matches found", vim.log.levels.WARN)
    return
  end

  if #results == 1 then
    open_result(results[1], repo_path)
    return
  end

  local items = {}
  for _, result in ipairs(results) do
    local symbol = result.symbol or "-"
    local label = string.format("%s:%d [%s] score=%.3f", result.file_path, result.start_line, symbol, result.score)
    table.insert(items, { label = label, result = result })
  end

  vim.ui.select(items, {
    prompt = "xtrc matches",
    format_item = function(item)
      return item.label
    end,
  }, function(choice)
    if choice then
      open_result(choice.result, repo_path)
    end
  end)
end

function M.jump(query)
  local q = query
  if q == nil or q == "" then
    q = vim.fn.input("xtrc query: ")
  end
  if q == nil or q == "" then
    return
  end

  local repo_path = resolve_repo()
  local payload = vim.fn.json_encode({
    repo_path = repo_path,
    query = q,
    top_k = M.config.top_k,
  })

  local cmd = {
    "curl",
    "-sS",
    "-X",
    "POST",
    M.config.server_url .. "/query",
    "-H",
    "Content-Type: application/json",
    "-d",
    payload,
  }

  local raw = vim.fn.system(cmd)
  if vim.v.shell_error ~= 0 then
    notify("Request failed. Is xtrc server running?", vim.log.levels.ERROR)
    return
  end

  local ok, response = pcall(vim.fn.json_decode, raw)
  if not ok then
    notify("Invalid JSON response from xtrc server", vim.log.levels.ERROR)
    return
  end

  if response.status == "error" then
    local msg = response.error and response.error.message or "unknown error"
    notify("xtrc error: " .. msg, vim.log.levels.ERROR)
    return
  end

  pick_and_open(response.results or {}, repo_path)
end

function M.setup(opts)
  M.config = vim.tbl_deep_extend("force", M.config, opts or {})

  vim.api.nvim_create_user_command("xtrc", function(params)
    M.jump(params.args)
  end, {
    nargs = "*",
    desc = "Semantic code jump via xtrc",
  })
end

return M
