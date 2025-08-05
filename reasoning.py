# reasoning.py - 100% offline reasoning & rephrasing (no APIs, no huggingface)
import re

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

positive_words = sorted(set([
    "great","excellent","amazing","awesome","fantastic","wonderful","perfect","superb","outstanding","impressive",
    "good","nice","love","loved","like","liked","satisfied","happy","pleased","delighted","thrilled","enjoyed",
    "reliable","fast","responsive","smooth","sturdy","durable","high-quality","premium","comfortable","accurate",
    "easy","simple","convenient","useful","affordable","worth","recommend","best","brilliant","solid","clean",
    "efficient","cool","beautiful","elegant","intuitive","helpful","supportive","timely","on-time","secure"
]))
negative_words = sorted(set([
    "bad","poor","terrible","awful","horrible","worst","disappointing","disappointed","unhappy","unpleasant",
    "hate","hated","useless","garbage","junk","cheap","flimsy","unreliable","faulty","defective","broken",
    "slow","laggy","buggy","glitchy","freezes","freezing","crash","crashes","crashing","noisy","weak","loose",
    "inaccurate","confusing","hard","difficult","complicated","unintuitive","dirty","damaged","late","delayed",
    "rude","unhelpful","unresponsive","ignored","cancelled","missing","lost","refund","refunds","returned","return",
    "warranty","waste","overpriced","expensive","pricey","cheaply","fake","misleading","deceptive"
]))

positive_phrases = sorted(set([
    "works well","works great","easy to use","worth the price","worth it","high quality","top notch",
    "on time","as described","better than expected","great value","fast delivery","well made"
]))
negative_phrases = sorted(set([
    "does not work","did not work","stopped working","quit working","hard to use","difficult to use",
    "waste of money","not worth it","poor quality","low quality","arrived late","arrived damaged",
    "false advertising","missing parts","no response","customer service","never again","would not recommend"
]))

def _find_matches(text: str, words, phrases):
    found = set()
    norm = _normalize(text)
    for w in words:
        if re.search(rf"\b{re.escape(w)}\b", norm):
            found.add(w)
    for p in phrases:
        if p in norm:
            found.add(p)
    return sorted(found)

def explain_reason(review: str, sentiment: str) -> str:
    pos = _find_matches(review, positive_words, positive_phrases)
    neg = _find_matches(review, negative_words, negative_phrases)
    if sentiment == "Negative":
        if neg:
            return "Negative sentiment indicated by cues such as: " + ", ".join(neg[:6]) + "."
        return "Negative tone detected (complaints, issues, or dissatisfaction)."
    if sentiment == "Positive":
        if pos:
            return "Positive sentiment indicated by cues such as: " + ", ".join(pos[:6]) + "."
        return "Positive tone detected (praise, satisfaction, or good experience)."
    if pos and neg:
        return "Mixed cues found (both favorable and critical terms), resulting in a neutral overall tone."
    return "Neutral sentiment detected (no strong positive or negative cues)."

soften_map = {
    "worst": "not ideal",
    "terrible": "unsatisfactory",
    "awful": "unsatisfactory",
    "horrible": "very disappointing",
    "hate": "strongly dislike",
    "useless": "not useful for my needs",
    "garbage": "very low quality",
    "junk": "very low quality",
    "disgusting": "unpleasant",
    "fake": "not genuine",
    "liar": "misleading",
    "cheat": "misleading",
    "broken": "not functioning",
    "faulty": "not functioning properly",
    "defective": "not functioning properly",
    "slow": "not fast",
    "laggy": "not very responsive",
    "buggy": "has some issues",
    "glitchy": "has glitches",
    "crash": "close unexpectedly",
    "crashes": "closes unexpectedly",
    "noisy": "loud",
    "weak": "not very strong",
    "cheap": "lower quality",
    "flimsy": "not sturdy",
    "rude": "not courteous",
    "unhelpful": "not helpful",
    "ignored": "did not receive a response",
    "scam": "potentially misleading",
    "liars": "misleading"
}

soften_phrases = [
    (r"\b(waste of money)\b", "not a good value for me"),
    (r"\b(does not work|did not work|doesn't work|didn't work)\b", "did not work as expected"),
    (r"\b(stopped working|quit working)\b", "stopped working as expected"),
    (r"\b(would not recommend|won't recommend|wouldn't recommend)\b", "I wouldn’t recommend this"),
    (r"\b(never again)\b", "I would prefer not to purchase this again"),
    (r"\b(customer service)\b", "customer support"),
    (r"\b(false advertising)\b", "misleading description")
]

def _soften_sentence_endings(text: str) -> str:
    return re.sub(r"!+", ".", text)

def _apply_phrase_softening(text: str) -> str:
    out = text
    for pattern, replacement in soften_phrases:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out

def _apply_word_softening(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    out_tokens = []
    for t in tokens:
        low = t.lower()
        if low in soften_map:
            repl = soften_map[low]
            if t[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            out_tokens.append(repl)
        else:
            out_tokens.append(t)
    return "".join(
        (("" if re.fullmatch(r"[^\w\s]", tok) else (" " if i>0 and not re.fullmatch(r"[^\w\s]", out_tokens[i-1]) else "")) + tok)
        for i, tok in enumerate(out_tokens)
    )

def _add_hedging(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return trimmed
    if re.search(r"\b(worst|terrible|awful|hate|garbage|junk|scam)\b", trimmed, re.IGNORECASE):
        return "From my experience, " + trimmed[0].lower() + trimmed[1:]
    return trimmed

def rephrase_review(review: str) -> str:
    s = _soften_sentence_endings(review)
    s = _apply_phrase_softening(s)
    s = _apply_word_softening(s)
    s = _add_hedging(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
