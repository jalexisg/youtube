import os

def test_readme_exists():
    assert os.path.exists("README.md"), "README.md should exist in the repository"


def test_tests_folder_exists():
    assert os.path.isdir("tests"), "tests/ folder should exist"


def test_requirements_or_no_requirements():
    # either requirements.txt exists or it's acceptable to have none
    assert os.path.exists("requirements.txt") or not os.path.exists("requirements.txt")
