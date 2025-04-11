def scale_size(size: tuple[int, int], max_size: int) -> tuple[int, int]:
    aspect = size[0] / size[1]
    if aspect > 1:
        width = max_size
        height = int(max_size / aspect)
    else:
        width = int(max_size * aspect)
        height = max_size
    return width, height
