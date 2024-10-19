from numpy.random import rand

n = 8

def sample_action():
    # return rand((8))
    a, b, c, d, h, i = rand(6)
    e = rand(1)
    if e<.2:
        return (1, 1, 0, 0, a, b, c, d)
    elif e<.4:
        return (1, 0, 0, 0, a, b, c, d)
    elif e<.6:
        return (1, .5, 0, 0, a, b, c, d)
    elif e<.8:
        return (.3, i, 0, 0, a, b, c, d)
    else:
        return (h, 0, 0, 0, a, b, c, d)
