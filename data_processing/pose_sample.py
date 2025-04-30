import numpy as np


class PoseSample(object):

  def __init__(self, name, landmarks, class_name, embedding):
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name
    self.embedding = embedding

  def to_dict(self):
    return {
      'name': self.name,
      'class_name': self.class_name,
      'landmarks': np.array(self.landmarks).flatten().tolist()
    }

  @classmethod
  def from_row(cls, row):
    name = row['name']
    class_name = row['class_name']
    landmarks = row['landmarks']
    embedding = row['embeddings']
    return cls(name, landmarks, class_name, embedding)
