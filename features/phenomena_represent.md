# General thoughts

In order to represent the variety of Spanish adequately, one important factors is to focus on the properties that really distinguish the different varieties from one another, i.e. it does not make sense to focus on the properties that are shared among many different regions but more on those that are unique to certain dialects (or wouldn't this be too specific?)

Major problem that could arise for distinction using the *voseo*: Spanish is a pro-drop language

Many of the peculiarities occur in vernacluar language, this could pose a problem

Group El Salvador, Guatemala & Honduras?

Southern Cone as a broad linguistic area?

Group the Antilles?

Evaluation has to take into account similarity of different dialects, e.g. Venezuela and the Antilles are sometimes more similar 


# Possible groupings of countries for simplified classification

- Antilles (Cuba, Puerto Rico, Dominican Republic)
- Middle Central America (El Salvador, Guatemala, Honduras)
- Andean region (Colombia, Ecuador, Peru, Chile (only Northern), Bolivia, Argentina (only Northwest))
- River Plate (Argentina, Uruguay, Paraguay (only parts))

:warning: grouping now based on Lipski 2012 (`geo.pdf`)

# Possibly good phenomena

## *voseo* (keep in mind that *vos* can be used with multiple types of verbal morphology)

**not used in:**

- Cuba
- Dominican Republic (?)
- Mexico (except some parts of Chiapas)
- Peru (some regions)
- Puerto Rico
- Spain

**used in:**

- Argentina
- Bolivia
- Chile
- Colombia (*usted* as familiar pronoun)
- Costa Rica (*usted* as familiar pronoun)
- Ecuador
- El Salvador
- Guatemala
- Honduras
- Nicaragua
- Panama (but on the decline & *usted* is more frequent)
- Paraguay
- Peru (some regions)
- Uruguay
- Venezuela

### How to model?

simple word count (*vos* vs. *tú*)? However, Spanish is a pro-drop language, so this may be misleading. Look for the according endings additionally? Since there are different types of verbal morphology, this could also be an indicator for specific regions.

- vos: pp-2p
- tú: ps
- usted: pp-2cs

look for lemma `vos`, `tú` and `usted` and the POS tags `pp-2p`, `ps` and `pp-2cs` respectively --> count!
! put subject and direct object pronouns together (should I maybe disambiguate?)

ALSO: take different voseo conjugation as feature too!
--> one feature with seven indices:

- 0: `-ái(s)`
- 1: `-éi(s)`
- 2: `-ís` for `-er`
- 3: `-á(s)`
- 4: `-é(s)`
- 5: `-a(s)`
- 6: `-e(s)`

:question: is it maybe too "risky" to incorporate the shortened versions (missing the `s`)? &rarr; check data!
&rarr; too risky!

check word after `vos` or second verb after `vos` if between there is `ADV`, `MISMO` &rarr; look for these specific endings
if the ending occurs: check POS tag &rarr; if verb, continue? maybe don't dismiss when pos tag is not verb!
if ending is `-ís` check whether lemma ends in `-er`, only then assign type B
if pos tag is verb but tells us about anything else than indicative present, dismiss
if pos tag is unclear, make sure there is no `í` (conditional, imperfect `-er`), `ar` (future `-ar`) or `as` (imperfect `-ar`) right before the ending 

:white_check_mark:


## clitic doubling:

- Argentina
- Bolivia
- Colombia
- Ecuador (with *le* as direct object clitic)
- Mexico
- Paraguay (with *le* and *les* as direct object clitics)
- Peru (but not when direct object has been fronted)

### How to model?

make use of POS tags, `po` should be the appropriate tag

:warning: might not occur in data

## double possessives

- Bolivia
- Colombia (Pacific coast)
- Honduras (occasionally)
- Mexico (some regions)

### How to model?

make use of POS tags, `dp-` should be the appropriate tag

:warning: might not occur in data

## use of overt subjects (some varieties use more pronouns than others)

### How to model?

make use of POS tags, simply let the number of subject pronouns be a predictor, `ps` is the appropriate tag

**BUT:** `ps` is also used for direct object pronouns &rarr; I have to disambiguate!

:question: How?

&rarr; check n tokens after pronoun, if there is a verb, assume that it belongs to the pronoun and count it!
This is not fool-proof because the verb could of course also belong to some other subject &rarr; check morphological annotation of POS tag

**Problem:** `ellos` has lemma `él`

Should I check for the appropriate person if it's a verb?
Would it be possible to create "confident" vs. "unsure" values for the features? E.g. a "confident" datapoint gets a score of 2, and an unsure one one of 1?

:white_check_mark:

## subjects before infinitives or gerunds

- Argentina
- Colombia
- Cuba
- Dominican Republic
- Ecuador
- Panama
- Puerto Rico
- Venezuela

### How to model?

look out for the sequence `ps` + `vr` and also `ps` + `vp(-00)` and use the count as predictor

:warning: this is tricky bc we might encounter examples like:  "y era importante para nosotros ganar hoy" where "nosotros" and "ganar" are actually separate from each other

:white_check_mark:


## non-inverted questions

- Cuba
- Dominican Republic
- Panama (Panama City)
- Puerto Rico
- Venezuela

### How to model?

look out for the sequence `WH-pronoun` + `pronoun` and use the count as predictor 
appropriate POS tags for the following pronouns are: `pd-3cs"`,`pd-3fp"`,`pd-3fs`,`pd-3mp`,`pd-3ms`,`ps`,`pp-1cs`,`pp-2cp`,`pp-2cs`,`pp-2p`,`fp`

also make sure that the question is not initiated with "por qué" because inversion is generally accepted in this case (source: https://doi.org/10.1075/avt.15.03baa)

:white_check_mark:

## "intensive" *ser*

- Colombia
- Ecuador
- Panama
- Venezuela

### How to model?

look out for `verb` + `ser`, (find out appropriate POS tags)

:warning: doesn't seem to occur in data

## `indefinite / indirect article + possessive`

- El Salvador
- Guatemala
- Honduras
- Mexico
- Paraguay

### How to model?

look out for the sequence `indefinite / indirect article` + `possessive`, (find out appropriate POS tags or simply use lemma)

appropriate grouped POS tags are `ARTI`, `PP`, `DETP` and `NN` and `NE` for the nouns

:white_check_mark:

## frequency of specific tenses (some varieties prefer specific tenses, e.g. Peruvian Spanish uses perfective instead of preterite a lot)

### How to model?

tenses seem to be annotated to a certain extend but I'm not sure which means which, an alternative approach could be to simply make use of the verbal morphology accompanied with the POS tag `v` to verify that the token is actually a verb and count the tenses this way. Perfective ofc would have to be counted using the sequence `v` + `vps-ms` and gerund with `form of estar` + `vps-ms` (*estar* can have different tags, depending on the person, e.g. `vip-2s` or `vip-3p`)

:white_check_mark:

## null direct object (correlated with clitic doubling of direct objects)

- Argentina
- Bolivia (?)
- Colombia (?)
- Ecuador
- Mexico (??????)
- Paraguay
- Peru

### How to model?

that's a tough one

## diminutives

- *-ico*
    - Colombia
    - Costa Rica
    - Cuba
- *-ito*
    - Mexico
- *-illo*
    - Mexico (Chiapas)
- *-ingo*
    - Bolivia

### How to model?

look out for the diminutive endings when the corresponding POS tag is actually a noun and further check whether the token is listed as a Spanish word in the spacy model, since there are many words that are "normal" words and end with one of the diminutive endings. If the word is not in a word list, it's more likely that it is a word transformed by a diminutive ending.

:white_check_mark:


## use of present subjunctive even if it is preceeded by a verb in past tense

- Argentina
- Ecuador
- Peru

### How to model?

…

:warning: doesn't seem to occur in the data


## use of *vosotros* to distinguish Spain Spanish

:white_check_mark: (put together with *vos*)


## *loísmo*, *laísmo*, *leísmo*

as described in [this paper](https://www.degruyter.com/document/doi/10.1515/9783110410327-004/html?lang=de), there are essentially three systems that deviate from the "correct" use of direct or indirect object clitics, namely the *loísmo*, the *laísmo* and the *leísmo*.

in *loísmo* and *laísmo*, the DO clitic *lo(s)* or *la(s)* is used instead of the IO clitic *le(s)*
in *leísmo*, the IO clitic *le(s)* is used instead of the DO clitics

According to Toscano Mateus (1953) (found in Lipski, p.251), South American Spanish generally has *loísta* tendencies, but some regions do not conform to this, e.g. Ecuador (*leísmo*) 

clitic pronouns: *lo, la, los, las, le, les*

:warning: *la, los, las* are not annotated as pronouns (`po`) but always as determiners (`ld-…`)

for the moment only count *lo, le, les*? &rarr; this would still capture the *leísmo* and *loísmo*, not *laísmo* though
one problem could be that this also captures pleonastic *lo* which is common for some dialects

:white_check_mark:


## use of *ser* vs. *estar*

some countries are associated with a more frequent use of *estar* (e.g. Mexico)

countries associated with less frequent *estar* usage:

- Peninsular Spanish
- Argentinian Spanish

this has also been supported by [this paper](https://www.researchgate.net/publication/335568928_Variability_in_serestar_Use_Across_Five_Spanish_Dialects_An_Experimental_Investigation)

### How to model?

as described in the paper mentioned above, context is key for the acceptability of *estar*, so it is very hard to do this "right"
one option is to simplify this and look for the number of occurrences for *ser* or *estar* respectively or only look at occurrences of *ser* & *estar* preceeding an adjective

:white_check_mark: (preceeding an adjective)

# Possibly unique phenomena

## Argentinia

- in vernacular speech of many regions, *yo* replaces *a mí* in dative-verb constructions
- postposed *eso* to include other members of a group :x:
- use of *grande* instead of *mucho*
- use of *ir en* to express motion  :x:

## Colombia

- in the Quechua-influenced highland region, diminiutive endings may be attached to clitic pronouns, especially in imperative constructions
- in the southern Andean region, verbal paraphrases with gerunds appear, e.g. combinations of venir + gerund but also others

## Costa Rica

- use of the suffix *-era* to refer to the act of doing something, items related to a general activity or a general collective
- in vernacular speech, collective nouns are often build by adding the suffix *-ada* :white_check_mark:

## Cuba

- *más* precedes negative words :white_check_mark:  
    - **combinations:** *más nunca*, *más nada*, *más nadie*

## Dominican Republic

- *ello* is used instead of the null expletive in existential and extraposed sentences, also common is *ello sí* / *no* as an answer to questions :x:

## Ecuador

- in vernacular speech, the diminutive suffix *-ito* is sometimes combined with demonstratives, numerals, interrogatives, gerunds, *no más* and more
- *estar* is often postposed, especially by Quechua-bilinguals
- throughout highland Ecuador, *dar* + gerund is often used as an imperative (possibly a calque from Quechua)

## Panama

- in vernacular speech, the addition of the suffix *vé* to familiar imperatives is quite common :x:

## Peru

- use of `object + verb` word order (from Quechua) ?
- combination of *muy* and *-ísimo* :white_check_mark:


# Overview: Implemented and Non-Implemented Features


:white_check_mark:

- **voseo**
    - **not used in:**
        - Cuba :white_check_mark:
        - Dominican Republic (?) :white_check_mark:
        - Mexico (except some parts of Chiapas) :white_check_mark:
        - Peru (some regions) :white_check_mark:
        - Puerto Rico :white_check_mark:
        - Spain :white_check_mark:
    - **used in:**
        - Argentina :white_check_mark:
        - Bolivia :part_alternation_mark: (*usted*)
        - Chile :x:
        - Colombia (*usted* as familiar pronoun) :part_alternation_mark: (:white_check_mark: for *usted*)
        - Costa Rica (*usted* as familiar pronoun) :part_alternation_mark: (:white_check_mark: for *usted*)
        - Ecuador :part_alternation_mark: (*usted*)
        - El Salvador :white_check_mark: (more *tú* than *vos* tho, *usted* used most)
        - Guatemala :white_check_mark: (more *tú* than *vos* tho, *usted* used most)
        - Honduras :part_alternation_mark: (*usted*)
        - Nicaragua :white_check_mark: (more *tú* than *vos* tho, *usted* used most)
        - Panama (but on the decline & *usted* is more frequent) :white_check_mark: (*usted*)
        - Paraguay :white_check_mark: (more *tú* than *vos* tho, *usted* used most)
        - Peru (some regions) :x:
        - Uruguay :white_check_mark: (more *tú* than *vos* tho, *usted* used most)
        - Venezuela :x:
    - overall, the trend seems to have gone towards a decline of the *voseo*, although it is still very present in e.g. AR, so it could prove to be a useful feature
    - some countries, such as ES and MX, prefer *tú* over *usted*, others, like *CR* or *PR*, vice versa
    - even though in many cases the results do not align with the literature, the fact that some countries show such clear tendencies in the data that were also described in the literature indicates that the feature was probably implemented correctly, especially given that the literature is quite "old" and already describes changing tendencies   
- **use of overt subjects**
    - there definitely seem to be different tendencies, however, there are not really any specific countries that stand out, the usefulness of this feature is thus questionable
- **subject before infinitive**
    - seems to happen in all other countries too (maybe because the feature is not actually good)
- **non-inverted questions**
    - result analysis finds this feature in three of the five countries, listed in the literature (also in others but to a much smaller extend):
        - Cuba :white_check_mark:
        - Dominican Republic :white_check_mark:
        - Panama (Panama City)
        - Puerto Rico :white_check_mark:
        - Venezuela
    - this feature seems to have been implemented correctly and it reflects to a big part what has been described in the literature
    - also, it could prove to be a helpful feature since it is indicative for three countries specifically
- **indefinite / indirect article + possessive**
    - result analysis finds this feature in two of the five countries, listed in the literature (also in others but to a much smaller extend):
        - El Salvador :white_check_mark:
        - Guatemala :white_check_mark:
        - Honduras
        - Mexico
        - Paraguay
    - this feature thus seems to have been implemented correctly and it reflects to a certain extend what has been described in the literature
    - also, it could prove to be a helpful feature, since it is indicative for two countries
- **frequencies of specific tenses**
    - the results reveal a different usage pattern for the different tenses, some of them could maybe even be indicative for a country, e.g. UY or ES
    - maybe overall too much unnecessary information, the distribution of some of the tenses is very similar
- **diminutives**
    - *-ito* is overall the most prevalent suffix, however, the distribution of the other diminutives is interesting, e.g. some dialects, like GT, prefer *-ico* over *-illo*, whereas others prefer *-illo* over *-ico* (e.g. CR, which actually goes against what is described in the literature). Furthermore, the analysis shows that BO is one of the few countries that actually exhibits a significant amount of occurrences of *ingo* as a suffix, which aligns with the literature
    - especially the case of CR leaves some doubt about whether the feature was implemented correctly
        - maybe the method using the oov function provided by spacy is not suitable for this?
- ***más* preceeding negative words (Cuba)**
    - this feature is found in CU but also in other countries (e.g. PA, UY, mostly in VE)
    - it is therefore not indicative for just one country, but the fact, that CU is one of the countries where it occurs to a bigger extend, indicates that the feature was probably implemented correctly
- **combination of *muy* and *-ísimo* (Peru)**
    - this feature seems to occur in many other countries, however, not at all in some (e.g. UY) and could therefore still prove to be a helpful feature
    - the fact that PE is still very high in the ranking, indicates that the feature was implemented correctly
- ***-ada* to form collective nouns (Costa Rica)**
    - this feature seems to occur in all other countries, especially in SV, so whether it will be actually helpful in distinguishing the different dialects is questionable
    - the fact that CR is still rather high in the ranking, indicates that the feature was probably implemented correctly
- **clitic pronouns**
    - *lo* is the most used clitic in all countries, however, such countries show more *loísta* tendencies than others, e.g. ES or UY
    - some countries definitely use more *le*
    - *les* is pretty similar in all countries
    - the usefulness of this feature is questionable, since the differences are most likely not significant enough
- ***ser* or *estar***
    - very sublte differences, probably not helpful as a feature


:warning:

- clitic doubling
- double possessives
- "intensive" *ser*
- use of present subjunctive even if it is preceeded by a verb in past tense
- *ello* instead of the null expletive in existential and extraposed sentences
- addition of the suffix *-vé* to familiar imperatives


# Overview: Which countries are actually reflected in the features such that they can be distinguished from the others?

- **AR**
    - voseo
- **BO**
    - diminutive (*-ingo*)
- **CL**
    - nada
- **CO**
    - (vosotros)
- **CR**
    - diminutives (*-illo*)
    - usted
    - (*-ada* suffix)
- **CU**
    - non-inverted questions
    - *más* preceeding negative words
    - (tuteo)
- **DO**
    - non-inverted questions
- **EC**
    - nada
- **ES**
    - vosotros
    - tuteo
    - (clitic pronoun (*lo*))
    - (tenses, e.g. very infrequent use of VIS)
- **GT**
    - indefinite article + possessive
    - (*-ada* words)
- **HN**
    - (tenses, e.g. very infrequent use of VIP)
- **MX**
    - tuteo
    - (vosotros)
- **NI**
    - muy + ísimo
- **PA**
    - (más + negative marker)
- **PE**
    - muy + ísimo
    - (tuteo)
- **PR**
    - non-inverted question
    - usted
    - (más + negative marker)
- **PY**
    -subject preceeding VPP
- **SV**
    - *-ada* suffix
    - indefinite article and possessive
    - *muy* preceeding *-ísimo*
- **UY**
    - very frequent use of multiple tenses (maybe just many verbs?)
    - *más* + negative marker
    - (clitic pronoun (*lo*))
- **VE**
    - *más* + negative marker
    - (tuteo)