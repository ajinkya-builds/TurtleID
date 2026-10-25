"""The GitHub Pages shell runs Stack v2 + MorphKey 100% in-browser by default ($0)."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mobile_page_defaults_to_free_browser_onnx():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "ajinkyapatil60/turtleid-student-onnx" in html
    assert "turtleid_student_int8.onnx" in html
    assert "./stack_v2_client.js" in html
    assert "./data/stack_v2_browser_bundle.json" in html
    assert "return BROWSER_MODE" in html
    assert "turtleid-api-wgbusgxhnq-el.a.run.app" not in html
    assert "return HF_SPACE" not in html
    assert "ajinkya-builds-turtleid.hf.space" not in html
    assert "URLSearchParams" in html
    assert ".get('api')" in html
    assert "localStorage" in html
    assert r"hf\.space" in html
    assert r"a\.run\.app" in html
    assert "All India / Unknown (Trade / Seizure)" in html
    assert 'id="state-select"' in html
    assert "pairwise_discriminator" in html
    assert "verification_questions" in html
    assert "CONFIRMED" in html
    assert "Schedule I" in html
    assert "How to tell #1 from #2" in html
    assert (ROOT / "docs" / "manifest.json").exists()
    assert (ROOT / "docs" / "sw.js").exists()
    assert (ROOT / "docs" / "stack_v2_client.js").exists()


def test_cloud_run_script_bakes_weights_and_listens_on_port():
    script = (ROOT / "scripts" / "deploy_cloudrun.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "PRELOAD_MODELS" in script
    assert "turtleid-api" in script
    assert "asia-south1" in script
    assert "gcloud builds submit" in script
    assert "--build-arg" in script
    assert "PORT=" not in script.split("set-env-vars", 1)[-1]
    assert "ARG PRELOAD_MODELS=0" in dockerfile
    assert "/app/models/hf_cache" in dockerfile
    assert "${PORT:-7860}" in dockerfile
    assert "python -m scripts.preload_models" in dockerfile


def test_modal_app_wraps_fastapi_on_a_t4():
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert "@modal.asgi_app()" in source
    assert 'gpu="T4"' in source
    assert "scaledown_window=120" in source
    assert "from api import app as api_app" in source
    assert "download_vision_weights" in source
    assert "requirements-cloudrun" not in source
    assert "cu124" in source
    assert "imageomics/bioclip-2.5-vith14" in source


def test_space_card_is_docker_on_7860():
    card = (ROOT / "spaces" / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in card
    assert "app_port: 7860" in card
    assert "startup_duration_timeout: 30m" in card
