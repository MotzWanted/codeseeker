import re


EndOfSentencePattern = re.compile(r"[.!?\n]+")
TitlePattern = re.compile(r"\b([A-Z][a-z]+)\.\s([A-Z][a-zA-Z]*)\b")  # Match titles like "Dr. Smith"
IsMarkdownHeaderPattern = re.compile(r"^(\s*#+)\s*+[a-zA-Z0-9\-'\"_\* ]{1,32}:?\s*$")
IsColonHeaderPattern = re.compile(r"^[a-zA-Z0-9\-'\"_\* ]{1,32}:\s*$")
IsHeaderRePattern = rf"({IsMarkdownHeaderPattern.pattern})|({IsColonHeaderPattern.pattern})"
IsInlinePunctuation = re.compile(r"(, )|(; )")
