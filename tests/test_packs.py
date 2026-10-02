import shutil
from pathlib import Path

import pytest
import yaml

from indicorderbench.packs import PackError, load_pack, validate_pack
from indicorderbench.schemas.scenario import CallerTurn, Language

FIX = Path(__file__).parent / "fixtures" / "minipack"


def test_load_minipack():
    pack = load_pack(FIX)
    assert pack.manifest.id == "minipack" and len(pack.scenarios) == 2
    assert pack.scenario("en_quantity_01").category.value == "quantity"
    assert pack.defaults_for(Language.EN_IN).closing.text == "That's all."
    assert len(pack.content_hash()) == 16
    assert [s.id for s in pack.filter(categories=["cancellation"])] == ["en_cancellation_01"]
    assert len(pack.filter(tags=["smoke"])) == 2 and pack.filter(tags=["nope"]) == []
    assert [s.id for s in pack.filter(ids=["en_quantity_01"])] == ["en_quantity_01"]
    assert pack.filter(languages=["hi-en"]) == []
    with pytest.raises(KeyError):
        pack.scenario("ghost")


def test_content_hash_changes_with_content(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    before = load_pack(tmp_path / "p").content_hash()
    assert before == load_pack(FIX).content_hash()
    path = tmp_path / "p" / "scenarios" / "en_quantity_01.yaml"
    path.write_text(path.read_text().replace("Two wraps", "Two wraps!"))
    assert load_pack(tmp_path / "p").content_hash() != before


def test_clip_path_convention(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    pack = load_pack(tmp_path / "p")
    turn = pack.scenario("en_quantity_01").caller.turns[0]
    assert pack.clip_path("en_quantity_01", turn, Language.EN_IN) is None
    clip = tmp_path / "p" / "clips" / "en_quantity_01" / "t1.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF")
    assert pack.clip_path("en_quantity_01", turn, Language.EN_IN) == clip
    # explicit audio field wins and is relative to the pack root
    explicit = CallerTurn(id="t9", text="x", audio="custom/x.wav")
    assert pack.clip_path("en_quantity_01", explicit, Language.EN_IN) is None
    (tmp_path / "p" / "custom").mkdir()
    (tmp_path / "p" / "custom" / "x.wav").write_bytes(b"RIFF")
    assert pack.clip_path("en_quantity_01", explicit, Language.EN_IN) == (
        tmp_path / "p" / "custom" / "x.wav"
    )
    # default turns resolve under clips/_defaults/<language>/<id>.wav
    closing = pack.defaults_for(Language.EN_IN).closing
    assert pack.clip_path("en_quantity_01", closing, Language.EN_IN) is None
    default_clip = tmp_path / "p" / "clips" / "_defaults" / "en-IN" / "closing.wav"
    default_clip.parent.mkdir(parents=True)
    default_clip.write_bytes(b"RIFF")
    assert pack.clip_path("en_quantity_01", closing, Language.EN_IN) == default_clip


def _with_explicit_audio(root: Path) -> Path:
    """Give en_quantity_01 an explicit clip on its first turn and on a clarification reply."""
    path = root / "scenarios" / "en_quantity_01.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["caller"]["turns"][0]["audio"] = "clips/does_not_exist.wav"
    doc["caller"]["clarifications"] = [
        {
            "id": "how_many",
            "match": ["how many"],
            "reply": {"text": "Two.", "audio": "clips/custom/two.wav"},
        }
    ]
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def test_validate_reports_missing_explicit_clips(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    path = _with_explicit_audio(tmp_path / "p")
    problems = validate_pack(tmp_path / "p")
    assert f"{path}: caller.turns[0].audio: missing clip clips/does_not_exist.wav" in problems
    assert (
        f"{path}: caller.clarifications[0].reply.audio: missing clip clips/custom/two.wav"
        in problems
    )
    assert len(problems) == 2
    with pytest.raises(PackError) as e:
        load_pack(tmp_path / "p")
    assert any("missing clip clips/does_not_exist.wav" in p for p in e.value.problems)
    for rel in ("clips/does_not_exist.wav", "clips/custom/two.wav"):
        clip = tmp_path / "p" / rel
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"RIFF")
    assert validate_pack(tmp_path / "p") == []
    turn = load_pack(tmp_path / "p").scenario("en_quantity_01").caller.turns[0]
    assert turn.audio == "clips/does_not_exist.wav"


def test_load_pack_can_skip_the_clip_check_for_synthesis(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    _with_explicit_audio(tmp_path / "p")
    pack = load_pack(tmp_path / "p", check_clips=False)
    assert pack.scenario("en_quantity_01").caller.turns[0].audio == "clips/does_not_exist.wav"


def test_validate_reports_unknown_item_and_option(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    bad = tmp_path / "p" / "scenarios" / "bad.yaml"
    doc = yaml.safe_load((FIX / "scenarios" / "en_quantity_01.yaml").read_text())
    doc["id"] = "en_quantity_99"
    doc["expected"][0]["orders"][0]["lines"][0]["item_id"] = "ghost"
    doc["expected"][0]["orders"][0]["lines"].append(
        {"item_id": "paneer_wrap", "quantity": 1, "modifiers": {"onion": "burnt"}}
    )
    doc["expected"][0]["orders"][0]["lines"].append(
        {"item_id": "mango_lassi", "quantity": 1, "modifiers": {"onion": "no_onion"}}
    )
    bad.write_text(yaml.safe_dump(doc))
    problems = validate_pack(tmp_path / "p")
    assert any("bad.yaml" in p and "ghost" in p for p in problems)
    assert any("bad.yaml" in p and "burnt" in p for p in problems)
    assert any("bad.yaml" in p and "not applicable" in p for p in problems)
    with pytest.raises(PackError) as e:
        load_pack(tmp_path / "p")
    assert e.value.problems == problems


def test_validate_rejects_duplicate_ids_and_wrong_menu(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    src = tmp_path / "p" / "scenarios" / "en_quantity_01.yaml"
    (tmp_path / "p" / "scenarios" / "dup.yaml").write_text(src.read_text())
    (tmp_path / "p" / "scenarios" / "wrong_menu.yaml").write_text(
        src.read_text()
        .replace("menu: mini", "menu: other")
        .replace("id: en_quantity_01", "id: en_quantity_02")
    )
    problems = validate_pack(tmp_path / "p")
    assert any("duplicate scenario id" in p for p in problems)
    assert any("wrong_menu.yaml" in p and "other" in p for p in problems)


def test_validate_reports_schema_errors_with_location(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    bad = tmp_path / "p" / "scenarios" / "bad.yaml"
    bad.write_text("id: bad\ntitle: t\nlanguage: xx\ncategory: quantity\nmenu: mini\n")
    problems = validate_pack(tmp_path / "p")
    assert any("bad.yaml" in p and "language" in p for p in problems)
    assert any("bad.yaml" in p and "caller" in p for p in problems)


def test_missing_manifest_and_missing_language_defaults(tmp_path: Path):
    assert validate_pack(tmp_path) == [f"{tmp_path / 'pack.yaml'}: missing"]
    shutil.copytree(FIX, tmp_path / "p")
    manifest = tmp_path / "p" / "pack.yaml"
    doc = yaml.safe_load(manifest.read_text())
    del doc["defaults"]["caller"]["en-IN"]
    manifest.write_text(yaml.safe_dump(doc))
    problems = validate_pack(tmp_path / "p")
    assert any("has no caller defaults" in p for p in problems)


def test_validate_ok_pack_returns_empty_list():
    assert validate_pack(FIX) == []


def test_clip_path_falls_back_to_language_defaults_for_shared_clarifications(tmp_path: Path):
    """Pack-default clarification replies are synthesised once per language, so a reply turn
    without a scenario-specific clip resolves under clips/_defaults/<language>/."""
    shutil.copytree(FIX, tmp_path / "p")
    pack = load_pack(tmp_path / "p")
    reply = CallerTurn(id="c_anything_else", text="No, that's all.")
    assert pack.clip_path("en_quantity_01", reply, Language.EN_IN) is None
    shared = tmp_path / "p" / "clips" / "_defaults" / "en-IN" / "c_anything_else.wav"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"RIFF")
    assert pack.clip_path("en_quantity_01", reply, Language.EN_IN) == shared
    # a scenario-specific clip with the same turn id wins over the shared one
    specific = tmp_path / "p" / "clips" / "en_quantity_01" / "c_anything_else.wav"
    specific.parent.mkdir(parents=True)
    specific.write_bytes(b"RIFF")
    assert pack.clip_path("en_quantity_01", reply, Language.EN_IN) == specific


def test_pack_root_is_absolute_even_when_loaded_from_a_relative_path(tmp_path: Path, monkeypatch):
    """Transcripts store clip paths derived from the root; they must survive a cwd change."""
    shutil.copytree(FIX, tmp_path / "p")
    monkeypatch.chdir(tmp_path)
    pack = load_pack(Path("p"))
    assert pack.root.is_absolute() and pack.root == (tmp_path / "p").resolve()
    clip = tmp_path / "p" / "clips" / "en_quantity_01" / "t1.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF")
    turn = pack.scenario("en_quantity_01").caller.turns[0]
    resolved = pack.clip_path("en_quantity_01", turn, Language.EN_IN)
    assert resolved is not None and resolved.is_absolute()


def test_scenario_level_caller_overrides_get_their_default_ids(tmp_path: Path):
    """closing/confirm/fallback/nudge overrides without an id are named like the defaults they
    replace, as soon as the scenario is loaded (synth and clip lookup rely on the id)."""
    shutil.copytree(FIX, tmp_path / "p")
    sc = tmp_path / "p" / "scenarios" / "en_quantity_01.yaml"
    sc.write_text(
        sc.read_text().replace(
            "caller:\n  turns:\n",
            'caller:\n  closing: {text: "That is it."}\n  nudge: {text: "Go on."}\n  turns:\n',
        )
    )
    s = load_pack(tmp_path / "p").scenario("en_quantity_01")
    assert s.caller.closing is not None and s.caller.closing.id == "closing"
    assert s.caller.nudge is not None and s.caller.nudge.id == "nudge"
    assert s.caller.confirm is None


def test_scripted_turn_ids_named_like_a_caller_default_are_rejected(tmp_path: Path):
    """A scripted turn called `closing` would share its clip path with the scenario's closing
    override, so those four ids are reserved."""
    shutil.copytree(FIX, tmp_path / "p")
    sc = tmp_path / "p" / "scenarios" / "en_quantity_01.yaml"
    sc.write_text(
        sc.read_text().replace('- text: "Two paneer', '- id: closing\n      text: "Two paneer')
    )
    with pytest.raises(PackError) as e:
        load_pack(tmp_path / "p")
    assert "reserved" in str(e.value) and "closing" in str(e.value)


def test_with_clips_resolves_every_turn_from_the_alternate_directory(tmp_path: Path):
    """`iob perturb` writes a drop-in replacement for <pack>/clips; Pack.with_clips points the
    clip lookup there without touching the pack on disk."""
    shutil.copytree(FIX, tmp_path / "p")
    noisy = tmp_path / "noisy"
    (noisy / "en_quantity_01").mkdir(parents=True)
    (noisy / "en_quantity_01" / "t1.wav").write_bytes(b"RIFFnoisy")
    (noisy / "_defaults" / "en-IN").mkdir(parents=True)
    (noisy / "_defaults" / "en-IN" / "closing.wav").write_bytes(b"RIFFnoisy")
    original = load_pack(tmp_path / "p")
    pack = original.with_clips(noisy)
    turn = pack.scenario("en_quantity_01").caller.turns[0]
    closing = pack.defaults_for(Language.EN_IN).closing
    assert (
        pack.clip_path("en_quantity_01", turn, Language.EN_IN)
        == noisy / "en_quantity_01" / "t1.wav"
    )
    assert pack.clip_path("en_quantity_01", closing, Language.EN_IN) == (
        noisy / "_defaults" / "en-IN" / "closing.wav"
    )
    assert pack.clips == noisy.resolve() and original.clips == tmp_path.resolve() / "p" / "clips"
    assert original.clip_path("en_quantity_01", turn, Language.EN_IN) is None
