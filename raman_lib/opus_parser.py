import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


class OpusParser(object):
    data: list | np.ndarray | pd.DataFrame
    params: list | pd.DataFrame
    metadata: bool | pd.DataFrame

    channel_dict = {
        "raman": 40
    }

    type_dict = {
        (0, 4): '<I',
        (1, 4): '<f',
        (1, 8): '<d'
    }

    def __init__(self, files, signal="raman", metadata=False):
        self.files = files
        self.signal = signal
        self.store_metadata = metadata

        # Prepare attributes
        self._bin_data = None
        self.params = []
        self.data = []

    def _validate_params(self):
        if isinstance(self.files, str):
            self.files = [self.files]
        self.files = [Path(file) for file in self.files]

        for file in self.files:
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist.")

        if self.signal not in self.channel_dict.keys():
            raise ValueError("Unknown signal type")

    # noinspection PyMethodMayBeStatic
    def _parse_header(self):
        header = self._bin_data[24:504]
        header = [header[i:i + 12] for i in range(0, len(header), 12)]

        chunks = []
        for chunk in header:
            if chunk == b'\x00' * 12:
                break
            chunks.append({"offset": struct.unpack('<I', chunk[-4:])[0],
                           "length": struct.unpack('<I', chunk[-8:-4])[0],
                           "block": struct.unpack('<B', chunk[0:1])[0],
                           "channel": struct.unpack('<B', chunk[1:2])[0],
                           "type": struct.unpack('<B', chunk[2:3])[0]})

        return pd.DataFrame(chunks)

    def _create_masks(self, chunks):

        data_mask = chunks.block == 15
        param_mask = chunks.block == 31
        acquisition_mask = chunks.block == 32
        optics_mask = chunks.block == 96
        sample_mask = chunks.block == 160
        channel_mask = chunks.channel == self.channel_dict[self.signal]

        self.data_chunk = chunks[data_mask & channel_mask].iloc[0]
        self.param_chunks = [
            chunks[param_mask & channel_mask].iloc[0],
            chunks[acquisition_mask].iloc[0],
            chunks[optics_mask].iloc[0],
            chunks[sample_mask].iloc[0]
        ]

    def _parse_param_block(self, offset, length, param_dict):
        param_bin = self._bin_data[offset:offset + length * 4]
        i = 0

        while i < len(param_bin):
            tag = param_bin[i:i + 3].decode('utf-8')
            if tag == 'END':
                break
            i += 4
            dtype = struct.unpack('<H', param_bin[i:i + 2])[0]
            length = struct.unpack('<H', param_bin[i + 2:i + 4])[0] * 2
            i += 4
            if dtype >= 2:
                content = param_bin[i:i + length].rstrip(b'\x00').decode('utf-8')
            else:
                content = struct.unpack(self.type_dict[dtype, length], param_bin[i:i + length])[0]
            param_dict[tag] = content
            i += length

    def _parse_param_blocks(self):
        params_tmp = {}
        for block in self.param_chunks:
            self._parse_param_block(block.offset, block.length, params_tmp)

        self.params.append(params_tmp)

    def _parse_data_block(self):
        offset = self.data_chunk.offset
        length = self.data_chunk.length
        data_bin = self._bin_data[offset:offset + length * 4]

        if not self.params:
            raise ValueError('Parameter list is empty. Was \'_parse_param_blocks\' executed first?')

        if len(data_bin) > (self.params[-1]['NPT'] + 1) * 4:
            data_tmp = self._parse_data_multiple(data_bin)
        else:
            data_tmp = self._parse_data_single(data_bin).reshape(1, -1)

        self.data.append(data_tmp)

    def _parse_data_single(self, data_bin):
        npt = self.params[-1]['NPT']
        if len(data_bin) > npt * 4:
            data_bin = data_bin[:4 * npt]
        return np.asarray(struct.unpack('<' + 'f' * npt, data_bin))

    def _parse_data_multiple(self, data_bin):
        header = struct.unpack('<' + 'I' * 4, data_bin[4:20])

        data = []
        ix = header[1]
        i = 0

        while i < header[0]:
            tmp = data_bin[ix:ix + header[2]]
            data.append(self._parse_data_single(tmp))
            ix += header[2] + header[3]
            i += 1
        return np.stack(data)

    def _clean_data(self):
        self.params = pd.DataFrame(self.params)
        reps = [len(array) for array in self.data]
        self.params = self.params.loc[self.params.index.repeat(reps)]

        index = pd.MultiIndex.from_arrays([np.repeat(self.files, reps),
                                           np.concatenate([np.arange(n) for n in reps])],
                                          names=['orig_file', 'spectrum_no'])
        self.params.index = index

        # Calculate wavenumbers and add to data
        wn_params = self.params.loc[:, ['NPT', 'FXV', 'LXV']].to_records(index=False)
        if np.any(wn_params != wn_params[0]):
            raise ValueError('One or more files use a different spectral range.')
        wn_params = wn_params[0]

        wns = np.linspace(wn_params[1], wn_params[2], wn_params[0])
        self.data = pd.DataFrame(np.row_stack(self.data), columns=wns, index=index)

        # Collect metadata
        self.metadata = self.params.loc[:, ['DAT', 'SNM', 'SFM', 'SRC', 'RLP', 'GRN', 'APT', 'INT', 'ASS']]
        self.metadata = self._format_metadata(self.metadata)

    @staticmethod
    def clean_string(s):
        s = re.sub(r'[^\w\s_]', '', s)

        s = re.sub(r'[\s._\-]+', '_', s)

        return s

    def _format_metadata(self, metadata):
        metadata.columns = ['date', 'sample_name', 'sample_form', 'laser', 'power', 'grating', 'aperture',
                            'integration_time', 'co_additions']

        metadata.date = pd.to_datetime(metadata.date)
        metadata.sample_name = [self.clean_string(s) for s in metadata.sample_name]
        metadata.sample_form = [self.clean_string(s) for s in metadata.sample_form]
        metadata.laser = [re.sub(r'\s', '', s) for s in metadata.laser]
        metadata.grating = [re.search(r', (\d+[a-z]),', s).group(1) for s in metadata.grating]
        metadata.aperture = [re.sub(r'\s', '', s) for s in metadata.aperture]

        return metadata

    @classmethod
    def from_dir(cls, path, signal='raman', metadata=False, recursive=False):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f'{str(path)} does not exist')
        if not path.is_dir():
            raise NotADirectoryError(f'{str(path)} is not a directory')

        opus_re = re.compile(r'.*\.\d+$')
        if recursive:
            files = [file for file in path.rglob('*.*[0-9]') if opus_re.match(str(file))]
        else:
            files = [file for file in path.glob('*.*[0-9]') if opus_re.match(str(file))]

        return cls(files, signal=signal, metadata=metadata)

    def parse(self):
        self._validate_params()

        for file in self.files:
            with open(file, 'rb') as f:
                self._bin_data = f.read()
            chunks = self._parse_header()
            self._create_masks(chunks)
            self._parse_param_blocks()
            self._parse_data_block()
        self._clean_data()

    def export_data(self, path, single=True, **kwargs):
        path = Path(path)
        if single:
            if path.exists() and not path.is_dir():
                raise NotADirectoryError(
                    f'Path is expected to be a directory when `single=True`, received {str(path)} instead')

            path.mkdir(parents=True, exist_ok=True)

            filename_counts = Counter()
            filenames_out = []
            for row1, row2 in zip(self.data.iterrows(), self.metadata.itertuples()):
                filename = '_'.join([row2.date.strftime('%y%m%d'), row2.sample_name, row2.sample_form])
                filename_full = filename + f'_{filename_counts[filename]:03}.csv'
                filenames_out.append(filename_full)
                filename_counts[filename] += 1

                row1[1].to_csv(path / filename_full, header=False, **kwargs)

            if self.store_metadata:
                # noinspection PyTypeChecker
                self.metadata.insert(0, 'file', filenames_out)
                self.metadata.reset_index(inplace=True)
                self.metadata.drop(columns='spectrum_no', inplace=True)
                self.metadata.set_index('file', inplace=True)
                self.metadata.to_csv(path / 'metadata.csv')


if __name__ == '__main__':
    parser = OpusParser.from_dir('/proj/raman/bernadette_rebekka/all/Blank', metadata=False, recursive=True)
    parser.parse()
    parser.export_data('/proj/raman')
