from flair.nn import Classifier
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from typing import List
from .util import Engine, Entity, adjust_offset, getLabelMapping, getLabel, normalize_offsets

standard_models = [
    "ner-fast",
    "ner",
    "ner-large",
    "ner-ontonotes-large",
    "de-ner",
    "de-ner-large"
]

class FlairEngine(Engine):
    """Uses flair for named entity recognition"""
    
    cachedClassifier: Classifier = None
    cachedClassifierName: str = None

    def __init__(self, logger):
        super().__init__(logger)

    def process(self, model: str, text: str) -> List[Entity]:
        (normText, normOffsets) = normalize_offsets(text)

        if self.cachedClassifierName == model:
            tagger = self.cachedClassifier
        else:
            self.logger.info(f"Loading classifier: {model}")
            tagger = Classifier.load(model)
            self.cachedClassifier = tagger
            self.cachedClassifierName = model
        
        splitter = SegtokSentenceSplitter()
        sentences = splitter.split(normText)

        self.logger.info(f"Processing {len(sentences)} sentences ...")
        tagger.predict(sentences)
        entities = []
        for sentence in sentences:
            spans = sentence.get_spans('ner')
            for span in spans:
                adjOffset = adjust_offset(
                    normOffsets, 
                    sentence.start_position + span.start_position, 
                    sentence.start_position + span.end_position
                )
                adjText = text[adjOffset[0]:adjOffset[1]]
                entType = getLabel(span.tag)
                if len(adjText) > 0 and entType is not None:
                    entities.append(Entity(text=adjText, type=entType, start=adjOffset[0]))

        return entities