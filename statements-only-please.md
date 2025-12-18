Extract standalone factual statements from the source text. Each output sentence must be fully interpretable in isolation.

Requirements for every output sentence:
- Explicitly name all entities (people, organizations, events, objects, times, places) in the same sentence; never rely on earlier text for identification.  
- Do not use any expression whose correct interpretation depends on previous sentences or omitted material. This includes:  
  - Personal, relative, and reflexive pronouns (e.g., "he," "she," "it," "they," "we," "you," "who," "which," "that," "himself," "themselves").  
  - Demonstratives and related forms (e.g., "this," "that," "these," "those," "such," "the former," "the latter," "the above," "the latter group," "this situation," "that event") unless the full referent is explicitly named in the same sentence.  
  - Definite descriptions (e.g., "the man," "the company," "the project," "the system," "the government," "the model," "the problem," evaluative epithets like "the idiot," "the hero") when they refer to something introduced only in previous discourse rather than fully specified in the same sentence.  
  - Verb phrase anaphora and ellipsis (e.g., "do so," "do it," "do that," "do the same," "did too," bare auxiliaries like "John left, and Mary did too," or omitted verbs or predicates recoverable only from context).  
  - Zero/omitted arguments or complements whose identity depends on context (e.g., dropped subjects/objects, null complements like "is ready Ø," "won Ø," when the missing material is not fully specified in the same sentence).  
  - Pro-forms such as "one," "ones," "so," "not so," or similar when they stand in for material only present in earlier text.  

Additional constraints:
- Do not use cross-sentence linking expressions like "therefore," "however," "in this case," "in that case," "for this reason," "as mentioned above."  
- Do not use cataphora (e.g., "In this experiment, the system..." where "this experiment" is only defined later).  
- Each sentence must contain all the lexical material needed to understand what it asserts, without looking at any other sentence.
