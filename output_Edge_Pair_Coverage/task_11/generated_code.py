def remove_Occ(s, ch):
    first = s.find(ch)
    last = s.rfind(ch)
    if first == -1 or last == -1:
        return s
    if first == last:
        return s[:first] + s[first+1:]
    else:
        return s[:first] + s[first+1:last] + s[last+1:]
