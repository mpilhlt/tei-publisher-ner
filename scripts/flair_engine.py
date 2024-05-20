from flair.nn import Classifier
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from typing import List
from .util import Engine, Entity, adjust_offset, getLabelMapping, normalize_offsets

class FlairEngine(Engine):

    def __init__(self, logger):
        super().__init__(logger)

    def process(self, model: str, text: str) -> List[Entity]:
        (normText, normOffsets) = normalize_offsets(text)
        
        tagger = Classifier.load(model)
        splitter = SegtokSentenceSplitter()
        sentences = splitter.split(normText)

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
                entities.append(Entity(text=adjText, type=span.tag, start=adjOffset[0]))

        return entities