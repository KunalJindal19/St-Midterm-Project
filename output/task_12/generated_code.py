def sort_matrix(M):
    def row_sum(row):
        return sum(row)
    return sorted(M, key=row_sum)
