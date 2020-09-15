from tensorpack.dataflow.format import LMDBData
from tensorpack.dataflow.serialize import LMDBSerializer
from tensorpack.dataflow.common import MapData, LocallyShuffleData
from tensorpack.utils.serialize import dumps, loads
import numpy
import itertools

class MyLMDBData(LMDBData):
  def __init__(self, lmdb_path, shuffle=True, keys=None, batch_from_disk=150):
    self.batch_from_disk = batch_from_disk
    super().__init__(lmdb_path, shuffle, keys)

  def __iter__(self):
    with self._guard:
      if not self._shuffle:
        c = self._txn.cursor()
        while c.next():
          k, v = c.item()
          if k != b'__keys__':
            yield [k, v]

      else:
        batches = self._size/self.batch_from_disk
        batched_keys = numpy.array_split(self.keys, batches)
        self.rng.shuffle(batched_keys)
        labels = []
        for k in batched_keys[0]:
          v = self._txn.get(k)
          a = loads(v)
          labels.append(a[1])
        print(labels)

        for k in itertools.chain.from_iterable(batched_keys):
          v = self._txn.get(k)
          yield [k, v]

class MyLMDBSerializer(LMDBSerializer):
  @staticmethod
  def load(path, shuffle=True, batch_from_disk=150):
    """
    This class reads the date in large chunks to optimize the HDD (yeah!) read times
    """
    df = MyLMDBData(path, shuffle=shuffle, batch_from_disk=batch_from_disk)
    return MapData(df, LMDBSerializer._deserialize_lmdb)

class MyLocallyShuffleData(LocallyShuffleData):
  def __init__(self, ds, buffer_size, num_reuse=1, shuffle_interval=None, nr_reuse=None):
    """
    This class locally shuflles data, in memory efficient way
    """
    super().__init__(ds, buffer_size, num_reuse, shuffle_interval, nr_reuse)
    self._inf_ds = ds

  def __iter__(self):
    with self._guard:
      for dp in self._inf_iter:
        self._iter_cnt = (self._iter_cnt + 1) % self.shuffle_interval
        # fill queue
        if self._iter_cnt == 0:
          self.rng.shuffle(self.q)
        for _ in range(self.num_reuse):
          if self.q.maxlen == len(self.q):
            yield self.q.popleft()
          self.q.append(dp)
      while len(self.q) > 0:
        yield self.q.popleft()