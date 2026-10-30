"""Lossless public text segmentation. Token intake is not usable-memory proof."""
import hashlib

def text_chunks(text,encode,*,budget=448):
    if not isinstance(text,str) or not text.strip() or type(budget)is not int or budget<1:
        raise ValueError('Nonempty source text and positive token budget required')
    start=0
    while start<len(text):
        # Bound tokenizer work to a local character window. Exact original
        # Unicode substrings are retained; token->text round trips are avoided.
        end=min(len(text),start+budget*4)
        if len(encode(text[start:end]))>budget:
            low,high=start+1,end
            while low<high:
                mid=(low+high+1)//2
                if len(encode(text[start:mid]))<=budget:low=mid
                else:high=mid-1
            end=low
        piece=text[start:end]
        if not piece or len(encode(piece))>budget:raise ValueError('A Unicode character exceeds the chunk token budget')
        yield dict(text=piece,start=start,end=end,tokens=len(encode(piece)))
        start=end

def source_identity(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
