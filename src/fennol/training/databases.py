
import numpy as np
from collections.abc import Iterable
import h5py

import io

try:
    import sqlite3

    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    def adapt_array(arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
except ImportError:
    sqlite3 = None




class DBDataset:
    def __init__(self,dbfile,table="training",select_keys=None):
        if sqlite3 is None:
            raise ImportError("sqlite3 is not available")
        self.con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        self.table = table
        self.keys = [k[1] for k in self.cur.execute(f"PRAGMA table_info({self.table})")]
        if select_keys is not None:
            self.select_keys = [k for k in self.keys if k in select_keys]
        else:
            self.select_keys = self.keys
        self.N = self.cur.execute(f"SELECT MAX(rowid) FROM {self.table}").fetchone()[0]

    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        if isinstance(idx,Iterable):
            indices = [i for i in idx]
        elif isinstance(idx,slice):
            indices = range(*idx.indices(self.N))
        else:
            indices = [idx]
        
        indices = [str(i+1) if i>=0 else str(i+self.N+1) for i in indices]
        query = f"SELECT {', '.join(self.select_keys)} FROM {self.table} WHERE rowid IN ({', '.join(indices)})"
        if len(indices)==1:
            data = self.cur.execute(query).fetchone()
            return {k:d for k,d in zip(self.select_keys,data)}
        else:
            return [{k:d for k,d in zip(self.select_keys,row)} for row in self.cur.execute(query)]

    def __iter__(self):
        for i in range(self.N):
            yield self[i]
    
    def __del__(self):
        self.con.close()
    
    def __repr__(self):
        return f"DBDataset({self.table})"
    
    def __str__(self):
        return f"DBDataset({self.table}) with {self.N} entries"
    
class H5Dataset:
    def __init__(self,h5file,table="training"):
        self.h5file = h5file
        self.table = table
        self.f = h5py.File(self.h5file,"r")
        self.dataset = self.f[self.table]
        self.N = len(self.dataset)
        self.keys = list(self.dataset["0"].keys())
    
    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        if isinstance(idx,Iterable):
            indices = [i for i in idx]
        elif isinstance(idx,slice):
            indices = range(*idx.indices(self.N))
        else:
            indices = [idx]
        
        indices = [str(i) for i in indices]
        if len(indices)==1:
            data = self.dataset[indices[0]]
            return {k:data[k][()] for k in self.keys}
        else:
            return [{k:data[k][()] for k in self.keys} for data in self.dataset[indices]]
    
    def __iter__(self):
        for i in range(self.N):
            yield self[i]
    
    def __del__(self):
        self.f.close()