MARKS = ".,;!?()"

def filter_marks(text: str) -> str:
    result = text
    for mark in MARKS:
        result = result.replace(mark, " ")
    return result
