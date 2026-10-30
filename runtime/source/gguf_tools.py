"""GGUF metadata packaging; neural tensor data is copied unchanged."""
import struct,shutil
def string(value):
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data

def key_string(name, value):
    return string(name) + struct.pack('<I', 8) + string(value)

def key_bytes(name, value):
    return string(name) + struct.pack('<IIQ', 9, 0, len(value)) + value

def repack(source, destination, replacements):
    if destination.exists():
        raise FileExistsError(destination)
    with source.open('rb') as file:
        fixed = file.read(24)
        magic, version, count, nkv = struct.unpack('<4sIQQ', fixed)
        if magic != b'GGUF':
            raise ValueError('Not GGUF')

        def number(fmt):
            return struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]

        def read_string():
            return file.read(number('<Q')).decode('utf-8')
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(kind):
            if kind in sizes:
                file.seek(sizes[kind], 1)
            elif kind == 8:
                file.seek(number('<Q'), 1)
            elif kind == 9:
                sub = number('<I')
                n = number('<Q')
                if sub in sizes:
                    file.seek(n * sizes[sub], 1)
                else:
                    for _ in range(n):
                        skip(sub)
            else:
                raise ValueError('Unsupported GGUF type ' + str(kind))
        entries = []
        present = set()
        alignment = 32
        for _ in range(nkv):
            start = file.tell()
            name = read_string()
            kind = number('<I')
            value_start = file.tell()
            if name == 'general.alignment':
                alignment = number('<I')
                file.seek(value_start)
            skip(kind)
            end = file.tell()
            if name in replacements:
                entries.append(replacements[name])
                present.add(name)
            else:
                file.seek(start)
                entries.append(file.read(end - start))
            file.seek(end)
        for name, value in replacements.items():
            if name not in present:
                entries.append(value)
        tensor_start = file.tell()
        for _ in range(count):
            read_string()
            dims = number('<I')
            file.seek(dims * 8 + 4 + 8, 1)
        tensor_end = file.tell()
        data_start = (tensor_end + alignment - 1) // alignment * alignment
        file.seek(tensor_start)
        tensor_info = file.read(tensor_end - tensor_start)
        with destination.open('xb') as output:
            output.write(struct.pack('<4sIQQ', magic, version, count, len(entries)))
            for value in entries:
                output.write(value)
            output.write(tensor_info)
            output.write(b' ' * (-output.tell() % alignment))
            file.seek(data_start)
            shutil.copyfileobj(file, output, 16 * 1024 * 1024)
    print('Packaged', destination.name, destination.stat().st_size, 'bytes;', count, 'language tensors; all existing metadata retained', flush=True)
