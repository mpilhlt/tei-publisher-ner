from typing import List
from .util import Engine, Entity, adjust_offset, getLabelMapping, normalize_offsets
from .cache import Cache
from logging import Logger

class SpacyEngine(Engine):

    cache: Cache

    def __init__(self, logger: Logger, cache: Cache):
        super().__init__(logger)
        
        self.cache = cache

    def process(self, model: str, text: str) -> List[Entity]:
        nlp = self.cache.getModel(model)

        if nlp is None:
            raise Exception(f"Model {model} not found")
        
        (normText, normOffsets) = normalize_offsets(text)
        doc = nlp(normText)

        labels = getLabelMapping(nlp.meta["labels"]["ner"])
        self.logger.info('Extracting entities using model %s', model)
        entities = []
        for ent in doc.ents:
            if ent.label_ in labels:
                adjOffset = adjust_offset(normOffsets, ent.start_char, ent.end_char)
                adjText = text[adjOffset[0]:adjOffset[1]]
                entities.append(Entity(text=adjText, type=labels[ent.label_], start=adjOffset[0]))
        return entities
