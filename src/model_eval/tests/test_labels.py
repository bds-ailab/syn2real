# test_cityscapes_labels.py
import types
import pytest

# Import the module under test.
# If your file is not named "cityscapes_labels.py", change the import below.
import model_eval.labels as mod


@pytest.mark.describe("Cityscapes Labels Definitions and Utilities")
class TestLabels:

    @pytest.mark.it(
        "should have Label namedtuple with correct fields and valid label entries"
    )
    def test_label_namedtuple_shape_and_types(self):
        expected_fields = (
            "name",
            "id",
            "trainId",
            "category",
            "categoryId",
            "hasInstances",
            "ignoreInEval",
            "color",
        )
        assert mod.Label._fields == expected_fields

        for lb in mod.labels:
            assert isinstance(lb, mod.Label)
            assert isinstance(lb.name, str) and lb.name != ""
            assert isinstance(lb.id, int)
            assert isinstance(lb.trainId, int)
            assert isinstance(lb.category, str) and lb.category != ""
            assert isinstance(lb.categoryId, int)
            assert isinstance(lb.hasInstances, bool)
            assert isinstance(lb.ignoreInEval, bool)
            assert isinstance(lb.color, tuple) and len(lb.color) == 3
            for c in lb.color:
                assert isinstance(c, int) and 0 <= c <= 255

    @pytest.mark.it("should have unique IDs and valid trainIds in range [0, 255]")
    def test_ids_and_trainIds_are_unique_where_expected(self):
        ids = [lb.id for lb in mod.labels]
        assert len(ids) == len(set(ids))

        train_ids = [lb.trainId for lb in mod.labels]
        for tid in train_ids:
            assert 0 <= tid <= 255

        names = [lb.name for lb in mod.labels]
        assert len(names) == len(set(names))

    @pytest.mark.it(
        "should have lookup dictionaries mapping correctly to Label objects"
    )
    def test_name2label_id2label_trainId2label_cover_all_entries(self):
        for lb in mod.labels:
            assert mod.name2label[lb.name] is lb
            assert mod.id2label[lb.id] is lb
            assert mod.trainId2label[lb.trainId].trainId == lb.trainId

    @pytest.mark.it("should group labels correctly by category")
    def test_category2labels_groups_consistently(self):
        seen = {id(lb): lb for lb in mod.labels}
        for cat, group in mod.category2labels.items():
            assert isinstance(cat, str) and cat != ""
            assert isinstance(group, list) and len(group) > 0
            for lb in group:
                assert id(lb) in seen
                assert lb.category == cat

    @pytest.mark.it(
        "should resolve assureSingleInstanceName correctly for various cases"
    )
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("car", "car"),
            ("cargroup", "car"),
            ("person", "person"),
            ("persongroup", "person"),
            ("sky", "sky"),
            ("skygroup", None),
            ("doesnotexist", None),
            ("doesnotexistgroup", None),
        ],
    )
    def test_assureSingleInstanceName(self, name, expected):
        result = mod.assureSingleInstanceName(name)
        assert result == expected

    @pytest.mark.it(
        "should return base name only if base label hasInstances=True when stripping 'group'"
    )
    def test_assureSingleInstanceName_consistency_with_hasInstances(self):
        for lb in mod.labels:
            grp = lb.name + "group"
            out = mod.assureSingleInstanceName(grp)
            if lb.hasInstances:
                assert out == lb.name
            else:
                assert out is None

    @pytest.mark.it(
        "should have valid Label entries in old_labels and reduced_labels lists"
    )
    def test_old_and_reduced_labels_are_well_formed(self):
        for attr in ("old_labels", "reduced_labels"):
            seq = getattr(mod, attr)
            assert isinstance(seq, list) and len(seq) > 0
            for lb in seq:
                assert isinstance(lb, mod.Label)
                assert isinstance(lb.name, str) and lb.name != ""
                assert isinstance(lb.id, int)
                assert isinstance(lb.trainId, int)
                assert isinstance(lb.category, str) and lb.category != ""
                assert isinstance(lb.categoryId, int)
                assert isinstance(lb.hasInstances, bool)
                assert isinstance(lb.ignoreInEval, bool)
                assert isinstance(lb.color, tuple) and len(lb.color) == 3
                for c in lb.color:
                    assert isinstance(c, int) and 0 <= c <= 255

    @pytest.mark.it("should have distinct colors for key classes")
    def test_color_values_are_rgb_and_distinct_for_key_classes(self):
        want_distinct = ["road", "sidewalk", "building", "sky", "car"]
        have = {
            n: mod.name2label[n].color for n in want_distinct if n in mod.name2label
        }
        vals = list(have.values())
        assert all(isinstance(c, tuple) and len(c) == 3 for c in vals)
        if len(vals) >= 2:
            assert len(set(vals)) == len(vals)

    @pytest.mark.it("should keep mapping examples from __main__ section consistent")
    def test_mapping_examples_in_comments_hold(self):
        if "car" in mod.name2label:
            assert mod.name2label["car"].id == mod.id2label[mod.name2label["car"].id].id

        some = next(iter(mod.id2label.values()))
        assert mod.id2label[some.id].category == some.category

        tid, lb = next(iter(mod.trainId2label.items()))
        assert lb.trainId == tid
