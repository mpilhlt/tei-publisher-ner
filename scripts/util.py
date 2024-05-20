from logging import Logger
import re
from typing import List, Tuple
from pydantic import BaseModel

# Mapping of NER pipeline labels to TEI Publisher labels
MAPPINGS = {
    "person": ("PER", "PERSON", "persName", "PRS"),
    "place": ("LOC", "GPE", "placeName", "geogName"),
    "organization": ("ORG", "orgName"),
    "author": ("AUT")
}

def getLabelMapping(nerLabels):
    """
    Returns a dictionary containing all pipeline entity labels which should be mapped to
    TEI Publisher annotation labels.
    """
    labels = {}
    for key in MAPPINGS:
        for nerLabel in nerLabels:
            if nerLabel in MAPPINGS[key]:
                labels[nerLabel] = key
    return labels

def getLabel(label):
    """
    Returns the TEI Publisher label for a given NER pipeline label.
    """
    for key in MAPPINGS:
        if label in MAPPINGS[key]:
            return key
    return None

def normalize_offsets(text: str) -> List:
    """
    Normalize the text by replacing sequences of 2 or more whitespace characters
    with a single space.

    Returns a tuple with 1) the normalized text and 2) a list of pairs, each
    containing a) the offset into the normalized text, b) the number of whitespace
    characters replaced up to the current offset
    """
    offsets = []
    offset = 0
    for match in re.finditer(r"[\s\n]{2,}", text):
        span = match.span()
        start = span[0] - offset
        offset += (span[1] - span[0] - 1)
        offsets.append((start, offset))
    return (re.sub(r"[\s\n]{2,}|\n", " ", text), offsets)

def adjust_offset(offsets: List, start: int, end: int) -> Tuple:
    """
    Recompute offsets into the normalized text to be relative to the original text.
    """
    i = 0
    # computed adjusted start
    while i < len(offsets) and offsets[i][0] < start: i += 1
    adjStart = start if i == 0 else offsets[i - 1][1] + start

    # computed adjusted end: entities may span across whitespace
    while i < len(offsets) and offsets[i][0] < end: i += 1
    adjEnd = end if i == 0 else offsets[i - 1][1] + end
    return (adjStart, adjEnd)

class Entity(BaseModel):
    """A single entity"""
    text: str
    type: str
    start: int

class Engine:
    """Abstraction around a named entity recognition engine"""

    logger: Logger

    def __init__(self, logger: Logger):
        self.logger = logger
    
    def process(self, model: str, text: str) -> List[Entity]:
        """Process the text using the given model"""
        raise NotImplementedError