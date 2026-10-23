from pathlib import Path

import pytest

from bibcut.identify import Identity, load_startlist, vote


def test_vote_sums_confidence_per_number() -> None:
    crops = [[("245", 0.9)], [("245", 0.8), ("45", 0.7)], [("45", 0.95)]]
    result = vote(7, crops, min_votes=1.5)
    assert result.number == "245"
    assert result.votes == (("245", pytest.approx(1.7)), ("45", pytest.approx(0.95)))


def test_vote_uses_the_best_reading_on_the_start_list() -> None:
    crops = [[("45", 0.95), ("245", 0.6)], [("245", 0.9)]]
    assert vote(1, crops, valid={"245"}, min_votes=1.5).number == "245"


def test_weak_votes_name_nobody() -> None:
    assert vote(3, [[("98", 0.9)]], min_votes=2.0) == Identity(3, None, (("98", 0.9),))


def test_no_readings() -> None:
    assert vote(3, [[], []]) == Identity(3, None, ())


def test_load_startlist(tmp_path: Path) -> None:
    path = tmp_path / "start.csv"
    path.write_text("number,name,categories\n7,Rider Seven,OW\n 12 ,,Open\n", encoding="utf-8")
    assert load_startlist(path) == {"7": "Rider Seven", "12": ""}


def test_startlist_needs_numbers(tmp_path: Path) -> None:
    path = tmp_path / "start.csv"
    path.write_text("name\nRider\n", encoding="utf-8")
    with pytest.raises(ValueError, match="number"):
        load_startlist(path)
