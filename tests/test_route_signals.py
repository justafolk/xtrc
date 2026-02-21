from xtrc.core.route_signals import extract_route_signal, infer_query_signal


def test_extract_route_signal_from_js_route() -> None:
    signal = extract_route_signal("router.post('/posts', createPostHandler)", symbol_name="createPost")

    assert signal is not None
    assert signal.method == "POST"
    assert signal.intent == "create"
    assert signal.resource == "post"
    assert "post" in signal.structural_terms
    assert "create" in signal.structural_terms


def test_extract_route_signal_from_python_decorator() -> None:
    signal = extract_route_signal("@router.delete('/posts/{post_id}')\ndef delete_post():\n    pass")

    assert signal is not None
    assert signal.method == "DELETE"
    assert signal.intent == "delete"
    assert signal.resource == "post"


def test_infer_query_signal_for_update_endpoint_query() -> None:
    signal = infer_query_signal("where is the PUT endpoint to update post")

    assert "update" in signal.intents
    assert "put" in signal.methods
    assert "post" in signal.structural_terms
