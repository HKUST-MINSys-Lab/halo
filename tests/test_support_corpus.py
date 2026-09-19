from types import SimpleNamespace

from training.support_classifier.corpus import open_vocabulary_holdout_labels
from training.tokenizer.pretrain_data import WindowKey


def test_open_vocabulary_holdout_is_global_deterministic_and_keeps_train_labels():
    labels = {label: index for index, label in enumerate(("a", "b", "c", "d", "e", "f"))}
    by_dataset = {"left": ("a", "b", "c", "d"), "right": ("c", "d", "e", "f")}
    refs = [SimpleNamespace(dataset=dataset) for dataset in by_dataset]
    train = []
    val = []
    for stream_i, dataset in enumerate(by_dataset):
        for window_i, label in enumerate(by_dataset[dataset]):
            train.append(WindowKey(stream_i, window_i, labels[label]))
            val.append(WindowKey(stream_i, window_i + 10, labels[label]))
    index = SimpleNamespace(label_ids=labels, refs=refs, train=train, val=val)

    first = open_vocabulary_holdout_labels(index, fraction=0.25, seed=17)
    second = open_vocabulary_holdout_labels(index, fraction=0.25, seed=17)
    assert first == second
    selected = set(first)
    for labels_for_dataset in by_dataset.values():
        assert len(selected & set(labels_for_dataset)) >= 2
        assert len(set(labels_for_dataset) - selected) >= 2

