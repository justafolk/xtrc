from xtrc.indexer.intent import extract_intent_metadata


def test_extract_intent_metadata_for_post_route() -> None:
    meta = extract_intent_metadata(
        file_path="routes/posts.js",
        symbol_kind="route",
        symbol="POST /posts",
        text="router.post('/posts', createPostHandler)",
    )

    assert meta.route_method == "POST"
    assert meta.route_path == "/posts"
    assert meta.route_intent == "create"
    assert meta.route_resource == "post"
    assert "create_resource" in meta.intent_tags
    assert "route_handler" in meta.intent_tags


def test_extract_intent_metadata_for_seed_script() -> None:
    meta = extract_intent_metadata(
        file_path="scripts/seeds/seed_posts.py",
        symbol_kind="function",
        symbol="seed_posts",
        text="def seed_posts(db):\n    db.insert({'title': 'x'})",
    )

    assert "seed_data" in meta.intent_tags
    assert "script" in meta.intent_tags
