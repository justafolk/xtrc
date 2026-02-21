local M = {}
local uv = vim.uv or vim.loop

M.config = {
  server_url = "http://127.0.0.1:8765",
  top_k = 8,
  repo_path = nil,
  debounce_ms = 50,
  use_telescope = true,
}

local function notify(msg, level)
  vim.notify(msg, level or vim.log.levels.INFO, { title = "xtrc" })
end

local function resolve_repo()
  return M.config.repo_path or vim.fn.getcwd()
end

local function resolve_file_path(file_path, repo_path)
  if string.sub(file_path, 1, 1) == "/" then
    return file_path
  end
  return repo_path .. "/" .. file_path
end

local function open_result(result, repo_path)
  local file_path = resolve_file_path(result.file_path, repo_path)

  vim.cmd("edit " .. vim.fn.fnameescape(file_path))
  local line = tonumber(result.start_line) or 1
  pcall(vim.api.nvim_win_set_cursor, 0, { line, 0 })
end

local function build_query_cmd(query, repo_path)
  local payload = vim.fn.json_encode({
    repo_path = repo_path,
    query = query,
    top_k = M.config.top_k,
  })

  return {
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
end

local function decode_query_response(raw)
  local ok, response = pcall(vim.fn.json_decode, raw)
  if not ok or type(response) ~= "table" then
    return nil, "Invalid JSON response from xtrc server"
  end

  if response.status == "error" then
    local msg = response.error and response.error.message or "unknown error"
    return nil, "xtrc error: " .. msg
  end

  return response.results or {}
end

local function query_sync(query, repo_path)
  local raw = vim.fn.system(build_query_cmd(query, repo_path))
  if vim.v.shell_error ~= 0 then
    return nil, "Request failed. Is xtrc server running?"
  end

  return decode_query_response(raw)
end

local function query_async(query, repo_path, on_done)
  local cmd = build_query_cmd(query, repo_path)
  if vim.system then
    vim.system(cmd, { text = true }, vim.schedule_wrap(function(obj)
      if obj.code ~= 0 then
        on_done(nil, "Request failed. Is xtrc server running?")
        return
      end

      local results, err = decode_query_response(obj.stdout or "")
      on_done(results, err)
    end))
    return
  end

  vim.schedule(function()
    local results, err = query_sync(query, repo_path)
    on_done(results, err)
  end)
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

local function search_with_telescope(default_query)
  local ok_pickers, pickers = pcall(require, "telescope.pickers")
  local ok_finders, finders = pcall(require, "telescope.finders")
  local ok_config, telescope_config = pcall(require, "telescope.config")
  local ok_actions, actions = pcall(require, "telescope.actions")
  local ok_action_state, action_state = pcall(require, "telescope.actions.state")
  local ok_sorters, sorters = pcall(require, "telescope.sorters")
  if not (ok_pickers and ok_finders and ok_config and ok_actions and ok_action_state and ok_sorters) then
    return false
  end

  local repo_path = resolve_repo()
  local conf = telescope_config.values
  local request_id = 0
  local picker

  local function entry_maker(result)
    local line = tonumber(result.start_line) or 1
    local symbol = result.symbol or "-"
    local score = tonumber(result.score) or 0
    local file_path = resolve_file_path(result.file_path, repo_path)

    return {
      value = result,
      ordinal = string.format("%s %d %s %.3f", result.file_path, line, symbol, score),
      display = string.format("%s:%d [%s] score=%.3f", result.file_path, line, symbol, score),
      filename = file_path,
      path = file_path,
      lnum = line,
      col = 1,
    }
  end

  local function refresh_picker(results)
    if not picker then
      return
    end
    picker:refresh(finders.new_table({
      results = results,
      entry_maker = entry_maker,
    }), { reset_prompt = false })
  end

  picker = pickers.new({}, {
    prompt_title = "xtrc",
    default_text = default_query or "",
    finder = finders.new_table({
      results = {},
      entry_maker = entry_maker,
    }),
    previewer = conf.grep_previewer({}),
    sorter = sorters.empty(),
    attach_mappings = function(prompt_bufnr)
      local timer = uv and uv.new_timer and uv.new_timer() or nil
      local augroup = vim.api.nvim_create_augroup("XtrcPicker" .. prompt_bufnr, { clear = true })
      local last_error

      local function cleanup()
        pcall(vim.api.nvim_del_augroup_by_id, augroup)
        if timer and not timer:is_closing() then
          timer:stop()
          timer:close()
        end
      end

      local function run_query()
        if not vim.api.nvim_buf_is_valid(prompt_bufnr) then
          return
        end
        local prompt = action_state.get_current_line() or ""
        request_id = request_id + 1
        local current_request_id = request_id

        if prompt == "" then
          last_error = nil
          refresh_picker({})
          return
        end

        query_async(prompt, repo_path, function(results, err)
          if current_request_id ~= request_id then
            return
          end

          if err then
            if err ~= last_error then
              notify(err, vim.log.levels.ERROR)
            end
            last_error = err
            refresh_picker({})
            return
          end

          last_error = nil
          refresh_picker(results or {})
        end)
      end

      local function schedule_query()
        if not timer then
          run_query()
          return
        end

        timer:stop()
        timer:start(M.config.debounce_ms, 0, vim.schedule_wrap(run_query))
      end

      vim.api.nvim_create_autocmd({ "TextChangedI", "TextChanged" }, {
        buffer = prompt_bufnr,
        group = augroup,
        callback = schedule_query,
      })
      vim.api.nvim_create_autocmd("BufWipeout", {
        buffer = prompt_bufnr,
        group = augroup,
        once = true,
        callback = cleanup,
      })

      actions.select_default:replace(function()
        local choice = action_state.get_selected_entry()
        actions.close(prompt_bufnr)
        if choice and choice.value then
          open_result(choice.value, repo_path)
        end
      end)

      schedule_query()
      return true
    end,
  })

  picker:find()
  return true
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
  local results, err = query_sync(q, repo_path)
  if err then
    notify(err, vim.log.levels.ERROR)
    return
  end

  pick_and_open(results, repo_path)
end

function M.search(query)
  if M.config.use_telescope and search_with_telescope(query) then
    return
  end
  M.jump(query)
end

function M.setup(opts)
  M.config = vim.tbl_deep_extend("force", M.config, opts or {})

  vim.api.nvim_create_user_command("Xtrc", function(params)
    M.search(params.args)
  end, {
    nargs = "*",
    desc = "Semantic code search via xtrc",
  })
end

return M
