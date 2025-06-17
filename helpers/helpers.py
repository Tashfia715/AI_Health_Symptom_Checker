def normalize_text(text: str) -> str:
    """
    Lowercase and strip user input.

    Parameters:
        text (str): Raw input.

    Returns:
        str: Normalized input.
    """
    return text.strip().lower()
