
import tqdm

class TqdmUploadFile:
    def init(self, file_obj, total, desc="Uploading"): 
        self.file_obj = file_obj 
        self.total = total 
        self.tqdm_bar = tqdm(total=total, desc=desc, unit='B', unit_scale=True)


    def read(self, n):
        data = self.file_obj.read(n)
        self.tqdm_bar.update(len(data))
        return data

    # Delegate attribute access to the underlying file object
    def __getattr__(self, attr):
        return getattr(self.file_obj, attr)