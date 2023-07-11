
def select_central_sites(n, m):
    '''Selects the `m` central contiguous sites
    from `n sites labeled `0, 1, ..., (n-1)`.

    In an even system size, that is when `n % 2 == 0`,
    the sites chosen are the smaller ones, that is closer
    to the left edge at `0` than to the right edge at `n-1`
    '''

    locations = tuple(range((n - m + 1)//2, (n + m + 1)//2))

    return locations
