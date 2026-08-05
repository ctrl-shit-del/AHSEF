from collections import Counter


class DatasetStatistics:

    @staticmethod
    def summarize(samples):

        emotions = Counter()

        splits = Counter()

        modalities = Counter()

        for sample in samples:

            emotions[sample.emotion] += 1

            splits[sample.split] += 1

            for modality in sample.modalities:

                modalities[modality] += 1

        return {

            "samples": len(samples),

            "emotion_distribution": dict(emotions),

            "split_distribution": dict(splits),

            "modalities": dict(modalities),

        }