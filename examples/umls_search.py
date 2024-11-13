import os
from tools import lexicon as um

apikey = os.environ.get("UMLS_API_KEY")
umls = um.UMLS(apikey)
# get top 10 search results
cui = umls.search("endometriosis", k=1)
synonyms = umls.synonyms(cui[0].ui)
print(synonyms)
# "CUI"|"Term"|"Code" default Term. CUI - the cui. Term - the text words. Code - the code.
