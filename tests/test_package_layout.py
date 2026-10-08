from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_public_repository_files_exist():
    expected = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "environment.yml",
        "aquagdr/models/udbf_depth_model.py",
        "aquagdr/train_fusion.py",
        "aquagdr/eval_fusion.py",
        "third_party/unifuse/LICENSE",
        "configs/stage1.yaml",
        "configs/stage2.yaml",
    ]
    missing = [name for name in expected if not (ROOT / name).is_file()]
    assert not missing, f"Missing expected release files: {missing}"


def test_configs_do_not_contain_machine_specific_absolute_paths():
    for name in ("stage1.yaml", "stage2.yaml"):
        text = (ROOT / "configs" / name).read_text(encoding="utf-8")
        assert "/data2/" not in text
        assert "D:\\" not in text
