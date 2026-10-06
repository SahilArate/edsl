"""Local runs must not touch the object store unless streaming is requested."""

from edsl import Model, QuestionFreeText
from edsl.object_store.store import ObjectStore
from edsl.runner import Runner
from edsl.runner.cas_integration import RunnerCASIntegration


def _make_job():
    q = QuestionFreeText(question_name="q", question_text="Say hi")
    return q.by(Model("test"))


def test_default_local_run_does_not_use_object_store(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default local run touched the object store")

    # Any access to the default store root fails the test
    monkeypatch.setattr(ObjectStore, "default_root", staticmethod(forbidden))
    # Constructing the CAS integration at all fails the test
    monkeypatch.setattr(RunnerCASIntegration, "__init__", forbidden)

    results = _make_job().run(
        cache=False, disable_remote_inference=True, progress_bar=False
    )

    assert len(results) == 1
    assert results[0]["answer"]["q"] is not None


def test_explicit_streaming_still_persists(monkeypatch, tmp_path):
    root = tmp_path / "objects"
    monkeypatch.setattr(ObjectStore, "default_root", staticmethod(lambda: root))

    runner = Runner()
    handle = runner.submit(_make_job(), cache=False, stream_to_cas=True)
    results = handle.results()

    cas = runner._job_cas[handle.job_id]
    assert cas.n_results == len(results)
    assert (root / cas.uuid).exists()
