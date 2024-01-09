def hassanat_distance(x, y):
    result = 0

    for xi, yi in zip(x, y):
        min_value = min(xi, yi)
        max_value = max(xi, yi)

        if min_value >= 0:
            result += 1 - (1 + min_value) / (1 + max_value)
        else:
            result += 1 - (1 + min_value + abs(min_value)) / (1 + max_value + abs(min_value))

    return result